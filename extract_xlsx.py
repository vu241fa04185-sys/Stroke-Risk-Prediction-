import zipfile
import xml.etree.ElementTree as ET
import csv

xlsx_path = "C:/Users/satya/.gemini/antigravity/brain/10e10973-db17-4a38-8a03-4d13e15f41d9/.user_uploaded/media_1792038152126_6ad198eb.xlsx"
csv_path = "C:/Users/satya/.gemini/antigravity/scratch/stroke_prediction_system/healthcare-dataset-stroke-data.csv"

with zipfile.ZipFile(xlsx_path, 'r') as z:
    # 1. Read shared strings
    shared_strings = []
    if 'xl/sharedStrings.xml' in z.namelist():
        tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
        ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        for si in tree.findall('ns:si', ns):
            t = si.find('ns:t', ns)
            if t is not None and t.text:
                shared_strings.append(t.text)
            else:
                # may have formatted runs
                text_parts = [r.find('ns:t', ns).text for r in si.findall('ns:r', ns) if r.find('ns:t', ns) is not None and r.find('ns:t', ns).text]
                shared_strings.append("".join(text_parts))

    # 2. Read sheet1
    sheet_tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

    rows_data = []
    for row in sheet_tree.findall('.//ns:row', ns):
        row_vals = []
        for cell in row.findall('ns:c', ns):
            cell_type = cell.attrib.get('t', '')
            val_elem = cell.find('ns:v', ns)
            if val_elem is not None and val_elem.text is not None:
                val = val_elem.text
                if cell_type == 's':
                    val = shared_strings[int(val)]
                row_vals.append(val)
            else:
                row_vals.append("")
        if any(row_vals):
            rows_data.append(row_vals)

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows_data)

print(f"Extracted {len(rows_data)} rows directly from XLSX to {csv_path}!")
