from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
from neu_det import parse_voc_annotation, draw_ground_truth_boxes, find_annotation_pair

from steel_core import (
    ROOT, KNOWN_CLASSES, UNKNOWN_CLASS, OUTPUT_DIR, INSPECTION_LOG,
    load_checkpoint, load_class_bank, load_manifest, image_path, load_image,
    extract_embedding, predict_embedding, save_class_bank, make_prototype,
    append_inspection_log,
)

def make_merged_vector(target_vector, source_vector, target_count, source_count):
    from steel_core import normalize_vector
    target = normalize_vector(target_vector)
    source = normalize_vector(source_vector)
    return normalize_vector(target * max(1, int(target_count)) + source * max(1, int(source_count))).tolist()

st.set_page_config(page_title="SteelDefectInspector", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.stApp {background:#0b1220;color:#e5e7eb}
[data-testid="stSidebar"] {background:#111c30}
div[data-testid="stMetric"] {background:#152238;padding:14px;border:1px solid #263854;border-radius:12px}
div[data-testid="stDataFrame"] {border:1px solid #263854;border-radius:8px}
</style>
""", unsafe_allow_html=True)
st.title("🔬 SteelDefectInspector")
st.caption("Open-set steel surface inspection | ResNet-18 embeddings | Operator-guided few-shot learning")

@st.cache_resource
def get_model():
    return load_checkpoint()

try:
    model, payload = get_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info("Run `python run_all.py --quick` after preparing the dataset.")
    st.stop()

threshold = float(payload["similarity_threshold"])
bank = load_class_bank()

st.sidebar.header("Inspection settings")
st.sidebar.success("Trained model loaded")
st.sidebar.metric("Known-class threshold", f"{threshold:.4f}")
st.sidebar.write("Known classes")
for name in KNOWN_CLASSES:
    st.sidebar.write("• " + name.replace("_", " ").title())
active_added = [name for name, entry in bank.get("classes", {}).items() if entry.get("status", "active") == "active"]
if active_added:
    st.sidebar.write("Additional active classes")
    for name in active_added:
        entry = bank["classes"][name]
        st.sidebar.write(f"• {name.replace('_', ' ').title()} (threshold {float(entry.get('threshold', threshold)):.3f})")

page = st.sidebar.radio("Navigate", ["Inspect Image", "NEU-DET Bounding Boxes", "Model Evaluation", "Few-Shot Learning", "Class Management", "Inspection Log", "About"])

if page == "Inspect Image":
    st.header("Steel Surface Inspection")
    st.write("Upload an image. Low similarity results are flagged for operator review rather than forced into a known class.")
    uploaded = st.file_uploader("Upload a defect image", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"], key="inspect_upload")
    shift = st.selectbox("Production shift", ["Day", "Evening", "Night"], key="inspect_shift")
    if uploaded is not None:
        try:
            image = load_image(uploaded)
            left, right = st.columns([1, 1])
            with left:
                st.subheader("Uploaded image")
                st.image(image, use_container_width=True)
            with st.spinner("Analyzing steel surface..."):
                embedding = extract_embedding(model, image)
                result = predict_embedding(embedding, payload, load_class_bank())
            with right:
                st.subheader("Inspection result")
                if result["label"] == "unknown":
                    st.warning("Unknown defect — operator review recommended")
                else:
                    st.success(f"Predicted class: {result['label'].replace('_', ' ').title()}")
                st.metric("Top similarity score", f"{float(result['confidence']):.4f}")
                st.metric("Applicable threshold", f"{float(result.get('threshold', threshold)):.4f}")
                st.caption(result["message"])
                if result["label"] == "unknown":
                    st.info("Record the operator's label and add five labelled examples in Few-Shot Learning.")
            st.subheader("Similarity scores")
            score_frame = pd.DataFrame([{"Class": k.replace("_", " ").title(), "Cosine similarity": float(v)} for k, v in result["scores"].items()]).sort_values("Cosine similarity", ascending=False)
            st.dataframe(score_frame, use_container_width=True, hide_index=True)
            st.bar_chart(score_frame.set_index("Class"))
            with st.form("save_inspection_form"):
                st.write("Save this inspection to the shift log")
                operator_label = st.text_input("Operator-confirmed label (optional)", value="")
                submitted = st.form_submit_button("Save inspection record", type="primary")
                if submitted:
                    append_inspection_log(uploaded.name, result, shift, operator_label.strip())
                    st.success("Inspection record saved.")
        except Exception as exc:
            st.error(f"Could not inspect this image: {exc}")
    else:
        st.info("Upload an image to begin inspection.")

elif page == "NEU-DET Bounding Boxes":
    st.header("NEU-DET Bounding-Box Viewer")
    st.write("Preview ground-truth boxes from NEU-DET Pascal VOC XML annotations. These are dataset annotations, not predictions from a trained detector.")
    st.info("Download NEU-DET separately. Upload one image and its matching XML annotation, or select files from a local NEU-DET folder below.")
    dataset_root = st.text_input("Optional local NEU-DET root folder", value=str(ROOT / "data" / "neu_det"))
    root_path = Path(dataset_root).expanduser()
    local_images = []
    if root_path.is_dir():
        image_dirs = [root_path / "IMAGES", root_path / "images", root_path / "JPEGImages", root_path]
        for folder in image_dirs:
            if folder.is_dir():
                local_images = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
                if local_images:
                    break
    if local_images:
        selected_path = st.selectbox("Select a local NEU-DET image", local_images, format_func=lambda p: p.name)
        ann_dirs = [root_path / "ANNOTATIONS", root_path / "annotations", root_path / "Annotations", root_path]
        ann_path = next((find_annotation_pair(selected_path, folder) for folder in ann_dirs if find_annotation_pair(selected_path, folder)), None)
        if ann_path:
            try:
                annotation = parse_voc_annotation(ann_path)
                with Image.open(selected_path) as im:
                    annotated = draw_ground_truth_boxes(im, annotation)
                st.image(annotated, caption=f"Ground-truth annotations: {ann_path.name}", use_container_width=True)
                st.write(f"**Objects:** {len(annotation['objects'])} | **Image size:** {annotation['width']} × {annotation['height']}")
                if annotation["objects"]:
                    st.dataframe(pd.DataFrame(annotation["objects"]), use_container_width=True, hide_index=True)
                else:
                    st.warning("The XML file contains no valid bounding boxes.")
            except Exception as exc:
                st.error(f"Could not parse annotation: {exc}")
        else:
            st.warning(f"No matching XML annotation found for {selected_path.name}. Expected {selected_path.stem}.xml in an annotations folder.")
    else:
        st.caption("No local NEU-DET image folder detected. Use the upload controls to preview an image and its annotation.")
    with st.expander("Upload an image and its matching XML annotation"):
        image_upload = st.file_uploader("NEU-DET image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="neu_det_image")
        xml_upload = st.file_uploader("Matching Pascal VOC XML", type=["xml"], key="neu_det_xml")
        if image_upload is not None and xml_upload is not None:
            try:
                annotation = parse_voc_annotation(xml_upload.getvalue().decode("utf-8", errors="replace"))
                annotated = draw_ground_truth_boxes(load_image(image_upload), annotation)
                st.image(annotated, caption="Ground-truth bounding boxes from uploaded XML", use_container_width=True)
                st.write(f"**Annotated objects:** {len(annotation['objects'])}")
                if annotation["objects"]:
                    st.dataframe(pd.DataFrame(annotation["objects"]), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Could not display NEU-DET annotations: {exc}")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    metrics_path = OUTPUT_DIR / "metrics.json"
    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    history_path = OUTPUT_DIR / "training_history.csv"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            evaluation = metrics.get("evaluation", {})
            report = evaluation.get("classification_report", {})
            macro = report.get("macro avg", {})
            # Read the actual validation accuracy from the training history.
            # The training pipeline currently records it as `val_acc`.
            validation_accuracy = None
            if history_path.exists():
                try:
                    validation_history = pd.read_csv(history_path)
                    validation_columns = ("val_acc", "validation_accuracy", "val_accuracy", "validation_acc")
                    validation_column = next(
                        (name for name in validation_columns if name in validation_history.columns),
                        None,
                    )
                    if validation_column and not validation_history.empty:
                        values = pd.to_numeric(validation_history[validation_column], errors="coerce").dropna()
                        if not values.empty:
                            validation_accuracy = float(values.iloc[-1])
                            # Handle either fractional (0.987) or percentage (98.7) values.
                            if validation_accuracy > 1.0:
                                validation_accuracy /= 100.0
                except (OSError, ValueError, pd.errors.ParserError):
                    validation_accuracy = None

            cols = st.columns(4)
            cards = [
                ("Validation accuracy", validation_accuracy, True),
                ("Known vs. unknown AUROC", evaluation.get("known_vs_unknown_auroc"), False),
                ("Macro precision", macro.get("precision"), True),
                ("Macro recall", macro.get("recall"), True),
            ]
            for col, (label, value, pct) in zip(cols, cards):
                if isinstance(value, (int, float)) and pd.notna(value):
                    display_value = f"{value:.2%}" if pct else f"{value:.3f}"
                else:
                    display_value = "N/A"
                col.metric(label, display_value)
            if validation_accuracy is None:
                st.info(
                    "Validation accuracy is not available because outputs/training_history.csv "
                    "does not contain a recognized validation-accuracy column. "
                    "Run the training pipeline to generate it."
                )
            st.subheader("Precision–Recall–F1 comparison")
            order = list(metrics.get("class_names", []))
            order += [label for label in evaluation.get("labels", []) if label not in order]
            order += [label for label in report if label not in order and label not in {"macro avg", "weighted avg"}]
            friendly = {"crazing":"Crazing", "inclusion":"Inclusion", "patches":"Patches", "pitted_surface":"Pitted Surface", "rolled_in_scale":"Rolled-in Scale", "unknown":"Unknown"}
            rows = []
            for label in order:
                item = report.get(label, {})
                if not isinstance(item, dict) or not any(k in item for k in ("precision", "recall", "f1-score")):
                    continue
                rows.append({"Defect class": friendly.get(label, label.replace("_", " ").title()), "Precision": f"{item['precision']:.1%}" if isinstance(item.get("precision"), (int,float)) else "—", "Recall": f"{item['recall']:.1%}" if isinstance(item.get("recall"), (int,float)) else "—", "F1-score": f"{item['f1-score']:.1%}" if isinstance(item.get("f1-score"), (int,float)) else "—", "Test images": int(item["support"]) if isinstance(item.get("support"), (int,float)) else "—"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            class_rows = [(label, report[label]) for label in order if label in report and label not in {"macro avg", "weighted avg"} and isinstance(report[label], dict) and isinstance(report[label].get("f1-score"), (int,float))]
            if class_rows:
                weak, values = min(class_rows, key=lambda x: x[1]["f1-score"])
                st.info(f"Key finding: {friendly.get(weak, weak.title())} has the lowest class-wise F1-score ({values['f1-score']:.1%}).")
            if matrix_path.exists():
                st.subheader("Confusion matrix")
                st.image(str(matrix_path), use_container_width=True)
            if history_path.exists():
                st.subheader("Training history")
                hist = pd.read_csv(history_path)
                st.dataframe(hist, use_container_width=True, hide_index=True)
                nums = [c for c in hist.columns if c != "epoch" and pd.api.types.is_numeric_dtype(hist[c])]
                if "epoch" in hist.columns and nums:
                    st.line_chart(hist.set_index("epoch")[nums])
            with st.expander("View raw evaluation JSON"):
                st.json(metrics)
        except Exception as exc:
            st.error(f"Could not read evaluation metrics: {exc}")
    else:
        st.warning("outputs/metrics.json was not found. Run the training pipeline first.")

elif page == "Few-Shot Learning":
    st.header("Operator-guided few-shot learning")
    st.write("Upload exactly five images confirmed by an operator to add a new defect class. The system averages their embeddings into a prototype; the neural network is not retrained.")
    st.warning("Use five labelled support examples only. Do not reuse evaluation/test images as support examples.")
    mode = st.radio("Example source", ["Upload five operator-labelled images", "Use reserved Scratches examples"], horizontal=True)
    if mode == "Upload five operator-labelled images":
        class_name = st.text_input("Operator-approved class name", placeholder="e.g. scratches, dent, corrosion").strip().lower().replace(" ", "_").replace("-", "_")
        labels_confirmed = st.checkbox("I confirm all five images belong to this defect class.")
        files = st.file_uploader("Select exactly five images", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"], accept_multiple_files=True, key="fewshot_uploads")
        if files:
            st.caption(f"Selected {len(files)} image(s). Exactly five are required.")
            previews = st.columns(min(5, len(files)))
            for i, f in enumerate(files[:5]):
                with previews[i % len(previews)]:
                    st.image(load_image(f), caption=f.name, use_container_width=True)
        replace_existing = st.checkbox("Replace an existing prototype with this new five-example set", value=False)
        if st.button("Create / update class prototype", type="primary"):
            try:
                if not class_name:
                    raise ValueError("Enter a class name.")
                if class_name in KNOWN_CLASSES and not replace_existing:
                    raise ValueError("This is one of the original known classes. Enable replacement only if you intend to update its prototype.")
                if not labels_confirmed:
                    raise ValueError("Confirm that the five images have the same operator-approved label.")
                if files is None or len(files) != 5:
                    raise ValueError("Select exactly five images.")
                current = load_class_bank()
                if class_name in current.get("classes", {}) and not replace_existing:
                    raise ValueError(f"Class '{class_name}' already exists. Enable replacement to update it.")
                imgs = [load_image(f) for f in files]
                prototype, new_threshold, support_scores = make_prototype(model, imgs)
                current.setdefault("classes", {})[class_name] = {
        "prototype": prototype.tolist(),
        "examples": [f.name for f in files],
        "num_examples": 5,
        "status": "active",
        "operator_approved": True,
        "threshold": new_threshold,
        "support_similarity_min": min(support_scores),
        "support_similarity_mean": sum(support_scores) / len(support_scores),
        "source": "operator_upload",
    }
                current.setdefault("history", []).append({"action": "add_or_update", "class_name": class_name, "num_examples": 5, "threshold": new_threshold, "source": "operator_upload"})
                save_class_bank(current)
                st.success(f"Class '{class_name}' is active. Its class-specific threshold is {new_threshold:.3f}; the neural network was not retrained.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        try:
            manifest = load_manifest()
            pool = manifest[manifest["split"] == "few_shot_pool"]
            st.metric("Reserved examples available", len(pool))
            if len(pool) == 5:
                st.dataframe(pool[["relative_path", "class", "split"]], use_container_width=True, hide_index=True)
                class_name = st.text_input("Reserved class name", value=UNKNOWN_CLASS).strip().lower().replace(" ", "_").replace("-", "_")
                replace_existing = st.checkbox("Replace existing prototype if this class is already active", key="reserved_replace")
                if st.button("Add reserved five-shot prototype", type="primary"):
                    try:
                        current = load_class_bank()
                        if class_name in KNOWN_CLASSES:
                            raise ValueError("Choose a new class name, not one of the five known classes.")
                        if class_name in current.get("classes", {}) and not replace_existing:
                            raise ValueError("Class already exists; enable replacement if intended.")
                        imgs = [load_image(image_path(row["relative_path"])) for _, row in pool.iterrows()]
                        prototype, new_threshold, support_scores = make_prototype(model, imgs)
                        current.setdefault("classes", {})[class_name] = {"prototype": prototype.tolist(), "examples": pool["relative_path"].tolist(), "num_examples": 5, "status": "active", "threshold": new_threshold, "support_similarity_min": min(support_scores), "support_similarity_mean": sum(support_scores)/len(support_scores), "source": "reserved_pool"}
                        current.setdefault("history", []).append({"action": "add_or_update", "class_name": class_name, "num_examples": 5, "threshold": new_threshold, "source": "reserved_pool"})
                        save_class_bank(current)
                        st.success(f"Prototype '{class_name}' added with threshold {new_threshold:.3f}. Network weights unchanged.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.error(f"Expected five reserved examples, found {len(pool)}.")
        except Exception as exc:
            st.error(f"Could not load reserved examples: {exc}")
    st.subheader("Active additional classes")
    current = load_class_bank()
    active = [{"Class": n, "Examples": v.get("num_examples", 0), "Threshold": round(float(v.get("threshold", threshold)), 4), "Status": v.get("status", "active"), "Source": v.get("source", "reserved pool / legacy")} for n, v in current.get("classes", {}).items() if v.get("status", "active") == "active"]
    if active:
        st.dataframe(pd.DataFrame(active), use_container_width=True, hide_index=True)
    else:
        st.info("No active additional classes yet.")

elif page == "Class Management":
    st.header("Class Management")
    st.write("Operator controls for rejecting a proposed class or merging an added class into another class.")
    current = load_class_bank()
    entries = current.get("classes", {})
    active = [n for n, v in entries.items() if v.get("status", "active") == "active"]
    if not active:
        st.info("No active additional classes are available to manage. Add one in Few-Shot Learning.")
    else:
        source = st.selectbox("Additional class to manage", active)
        entry = entries[source]
        st.write(f"Examples: {entry.get('num_examples', 0)} | threshold: {float(entry.get('threshold', threshold)):.3f} | source: {entry.get('source', 'legacy')}")
        left, right = st.columns(2)
        with left:
            if st.button("Reject / deactivate this class", type="secondary"):
                entries[source]["status"] = "rejected"
                current.setdefault("history", []).append({"action": "reject", "class_name": source})
                save_class_bank(current)
                st.success(f"Class '{source}' rejected. It will no longer be used for predictions; history is retained.")
                st.rerun()
        with right:
            targets = [n for n in KNOWN_CLASSES if n != source] + [n for n in active if n != source]
            target = st.selectbox("Merge into target class", targets, key="merge_target")
            if st.button("Merge class into target"):
                try:
                    source_vec = entries[source]["prototype"]
                    if target in KNOWN_CLASSES:
                        target_vec = current.get("overrides", {}).get(target, payload["centroids"][target])
                        current.setdefault("overrides", {})[target] = make_merged_vector(target_vec, source_vec, 210, int(entries[source].get("num_examples", 5)))
                    else:
                        target_entry = entries[target]
                        target_entry["prototype"] = make_merged_vector(target_entry["prototype"], source_vec, int(target_entry.get("num_examples", 5)), int(entries[source].get("num_examples", 5)))
                        target_entry["num_examples"] = int(target_entry.get("num_examples", 5)) + int(entries[source].get("num_examples", 5))
                    entries[source]["status"] = "merged"
                    entries[source]["merged_into"] = target
                    current.setdefault("history", []).append({"action": "merge", "class_name": source, "target": target})
                    save_class_bank(current)
                    st.success(f"Merged '{source}' into '{target}'. The source class is no longer independently predicted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Merge failed: {exc}")
    with st.expander("Class action history"):
        st.dataframe(pd.DataFrame(current.get("history", [])), use_container_width=True, hide_index=True)

elif page == "Inspection Log":
    st.header("Inspection Log")
    st.write("Inspection records are saved locally when an operator selects **Save inspection record** on the Inspect Image page.")
    if INSPECTION_LOG.exists():
        log = pd.read_csv(INSPECTION_LOG)
        if not log.empty:
            log["timestamp_local"] = log["timestamp_local"].astype(str)
            a, b, c = st.columns(3)
            a.metric("Recorded inspections", len(log))
            b.metric("Flagged for review", int((log["predicted_class"] == "unknown").sum()))
            c.metric("Active shifts", int(log["shift"].nunique()))
            st.subheader("Defect counts by shift")
            counts = log.groupby(["shift", "predicted_class"]).size().reset_index(name="count")
            st.dataframe(counts, use_container_width=True, hide_index=True)
            st.bar_chart(counts.pivot(index="shift", columns="predicted_class", values="count").fillna(0))
            st.subheader("Inspection records")
            st.dataframe(log.sort_values("timestamp_local", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download inspection log CSV", data=log.to_csv(index=False).encode("utf-8"), file_name="inspection_log.csv", mime="text/csv")
        else:
            st.info("The inspection log is empty.")
    else:
        st.info("No inspections have been saved yet.")

elif page == "About":
    st.header("About SteelDefectInspector")
    st.markdown("""
**Purpose:** Classify five known steel surface defects, flag unfamiliar images, and let an operator add a new defect class using five labelled examples.

**Known classes:** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale.  
**Held-out class:** Scratches. It is excluded from initial training and is treated as unknown until an operator adds its prototype.

**Unknown detection:** Cosine similarity between a ResNet-18 image embedding and class prototypes, compared with a calibrated known-class threshold or class-specific threshold for operator-added prototypes.

**Few-shot learning:** Exactly five examples are embedded and averaged into a prototype. The neural network is not retrained.

**Operator controls:** Save inspection records, deactivate/reject an added class, or merge it into another class.  
**NEU-DET:** The NEU-DET Bounding-Box Viewer reads Pascal VOC XML ground-truth annotations and draws labelled boxes. This is annotation visualization, not model-generated detection. A trained detector requires the NEU-DET dataset and separate detector training/evaluation. Similarity is not a calibrated probability.
""")

st.divider()
st.caption("SteelDefectInspector | Research prototype | Verify predictions against labelled inspection data.")

