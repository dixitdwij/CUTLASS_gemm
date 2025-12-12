import re
import json
import sys

def parse_ncu_log(log_content: str) -> dict:
    # Parses the ncu log content and returns json

    data = {}
    current_section = None
    
    # Track last message type (INF or OPT) to handle multi-line messages
    last_msg_type = None 

    # regex section headers 
    section_pattern = re.compile(r"^\s*Section:\s+(.*)")
    
    column_split_pattern = re.compile(r"\s{2,}")

    lines = log_content.splitlines()
    
    for line in lines:
        stripped_line = line.strip()
        
        # Emprty line
        if not stripped_line or set(stripped_line) <= set("- "):
            continue

        # New section detect
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1).strip()
            data[current_section] = {}
            last_msg_type = None
            continue

        if current_section:
            
            # tabel header
            if "Metric Name" in stripped_line and "Metric Unit" in stripped_line:
                last_msg_type = None
                continue

            # INF OPT msg
            if stripped_line.startswith("INF") or stripped_line.startswith("OPT"):
                msg_type = stripped_line[:3] # "INF" or "OPT" hopefully
                msg_content = stripped_line[3:].strip()
                
                # init or append (if multiple distinct blocks exist, concat)
                if msg_type in data[current_section]:
                    data[current_section][msg_type] += " " + msg_content
                else:
                    data[current_section][msg_type] = msg_content
                
                last_msg_type = msg_type
                continue

            # Process Metrics vs Message Continuation
            # Split line by whitespace gaps
            parts = column_split_pattern.split(stripped_line)

            if len(parts) >= 2:
                # It is likely a Metric row: [Name, Unit, Value] or [Name, Value]
                last_msg_type = None # Reset msg mode
                
                metric_name = parts[0].strip()
                metric_val_str = parts[-1].strip()
                metric_unit = parts[1].strip() if len(parts) > 2 else ""

                try:
                    clean_val = metric_val_str.replace(',', '') # Remove commas for large numbers
                    val = float(clean_val)
                    if val.is_integer():
                        val = int(val)
                except ValueError:
                    # Keep as string (e.g., "PolicySpread")
                    val = metric_val_str

                data[current_section][metric_name] = {
                    "val": val,
                    "unit": metric_unit
                }
            
            else:
                # If doesn't split into columns, it might be a continuation of INF/OPT
                if last_msg_type and last_msg_type in data[current_section]:
                    # Append line to prev msg
                    data[current_section][last_msg_type] += " " + stripped_line

    return data
    # return json.dumps(data, indent=4)

import os
if __name__ == "__main__":
    INPUT_FILE = "ncu_temp.txt"
    OUTPUT_FILE = "ncu_temp.json"
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        print(f"Reading from {INPUT_FILE}...")
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = parse_ncu_log(content)

            # print(json.dumps(data, indent=4))
            print(f'#####{data['GPU Speed Of Light Throughput']['Duration']['val']}#####')
            
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(json.dumps(data, indent=4), f, indent=4)
                
            print(f"Success! JSON data written to {OUTPUT_FILE}")
            
        except Exception as e:
            print(f"An error occurred: {e}")