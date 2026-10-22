from PIL import Image
from neu_det import parse_voc_annotation, draw_ground_truth_boxes


def test_parse_pascal_voc_annotation_and_draw_box():
    xml = '''<annotation><filename>sample.jpg</filename><size><width>100</width><height>80</height><depth>3</depth></size><object><name>scratches</name><bndbox><xmin>10</xmin><ymin>12</ymin><xmax>70</xmax><ymax>60</ymax></bndbox></object></annotation>'''
    parsed = parse_voc_annotation(xml)
    assert parsed["filename"] == "sample.jpg"
    assert parsed["width"] == 100 and parsed["height"] == 80
    assert len(parsed["objects"]) == 1
    assert parsed["objects"][0]["class"] == "scratches"
    out = draw_ground_truth_boxes(Image.new("L", (100, 80), 128), parsed)
    assert out.mode == "RGB" and out.size == (100, 80)
