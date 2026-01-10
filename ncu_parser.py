import re
import json
import sys

def parse_ncu_log(log_content: str) -> dict:

    data = {}
    current_section = None
    
    last_msg_type = None 

    section_pattern = re.compile(r"^\s*Section:\s+(.*)")
    
    column_split_pattern = re.compile(r"\s{2,}")

    lines = log_content.splitlines()
    
    for line in lines:
        stripped_line = line.strip()
        
        if not stripped_line or set(stripped_line) <= set("- "):
            continue

        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1).strip()
            data[current_section] = {}
            last_msg_type = None
            continue

        if current_section:
            
            if "Metric Name" in stripped_line and "Metric Unit" in stripped_line:
                last_msg_type = None
                continue

            if stripped_line.startswith("INF") or stripped_line.startswith("OPT"):
                msg_type = stripped_line[:3] # "INF" or "OPT" hopefully
                msg_content = stripped_line[3:].strip()
                
                if msg_type in data[current_section]:
                    data[current_section][msg_type] += " " + msg_content
                else:
                    data[current_section][msg_type] = msg_content
                
                last_msg_type = msg_type
                continue

            parts = column_split_pattern.split(stripped_line)

            if len(parts) >= 2:
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
                    val = metric_val_str

                data[current_section][metric_name] = {
                    "val": val,
                    "unit": metric_unit
                }
            
            else:
                if last_msg_type and last_msg_type in data[current_section]:
                    data[current_section][last_msg_type] += " " + stripped_line

    return data

class KernelPerformance:
    def __init__(self, parsed_data: dict):
        self.parsed_data = parsed_data
        
        def get_val(section, metric, default=0.0):
            try:
                return parsed_data.get(section, {}).get(metric, {}).get('val', default)
            except:
                return default

        self.duration_ms: float = get_val('GPU Speed Of Light Throughput', 'Duration')
        self.mem_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'Memory Throughput')
        self.sm__pct: float = get_val('GPU Speed Of Light Throughput', 'Compute (SM) Throughput')
        self.dram_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'DRAM Throughput')
        self.l1_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'L1/TEX Cache Throughput')
        self.l2_throughput_pct: float = get_val('GPU Speed Of Light Throughput', 'L2 Cache Throughput')
        
        self.ipc: float = get_val('Compute Workload Analysis', 'Executed Ipc Active')
        
        self.mem_max_bandwidth: float = get_val('Memory Workload Analysis', 'Max Bandwidth')
        self.l1_tex_hit_rate_pct: float = get_val('Memory Workload Analysis', 'L1/TEX Hit Rate')
        self.l2_hit_rate_pct: float = get_val('Memory Workload Analysis', 'L2 Hit Rate')
        
        self.reg_per_thread: int = int(get_val('Launch Statistics', 'Registers Per Thread', 0))

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
            
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4) # Fixed bug in original main block
                
            print(f"Success! JSON data written to {OUTPUT_FILE}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
