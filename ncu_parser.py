import json
import re

class NcuLogParser:
    def __init__(self):
        # Regex patterns
        self.header_launch_pattern = re.compile(
            r'\(([\d, ]+)\)x\(([\d, ]+)\),\s+Context\s+(\d+),\s+Stream\s+(\d+),\s+Device\s+(\d+),\s+CC\s+([\d\.]+)'
        )
        self.section_header_pattern = re.compile(r'Section:\s+(.*)')
        self.msg_pattern = re.compile(r'^\s*(INF|OPT)\s+(.*)')
        # Matches rows with at least 2 columns separated by 2+ spaces
        self.table_row_split_pattern = re.compile(r'\s{2,}')

    def parse_value(self, val_str):
        """Attempts to convert strings to int or float, otherwise returns string."""
        val_str = val_str.strip()
        try:
            return int(val_str)
        except ValueError:
            try:
                return float(val_str)
            except ValueError:
                return val_str

    def parse_header(self, line):
        """Parses the kernel signature and launch configuration."""
        result = {}
        
        if "(Params)" in line:
            sig_part, config_part = line.split("(Params)", 1)
            result['kernel_name'] = sig_part.strip()
        else:
            result['kernel_name'] = line
            config_part = ""

        match = self.header_launch_pattern.search(config_part)
        if match:
            grid = [int(x) for x in match.group(1).split(',')]
            block = [int(x) for x in match.group(2).split(',')]
            result['launch_config'] = {
                'grid_dim': grid,
                'block_dim': block,
                'context': int(match.group(3)),
                'stream': int(match.group(4)),
                'device': int(match.group(5)),
                'compute_capability': float(match.group(6))
            }
        return result

    def parse(self, raw_text):
        lines = [line.rstrip() for line in raw_text.strip().split('\n')]
        
        output = {
            "kernel_info": {},
            "sections": {},
            "messages": []
        }

        if lines:
            output["kernel_info"] = self.parse_header(lines[0])

        current_section = None
        current_msg_type = None
        
        in_table = False
        
        for i, line in enumerate(lines[1:], 1):
            stripped = line.strip()
            
            sec_match = self.section_header_pattern.search(line)
            if sec_match:
                current_section = sec_match.group(1).strip()
                output["sections"][current_section] = []
                in_table = False
                current_msg_type = None
                continue

            if stripped.startswith('---') or stripped.startswith('Metric Name'):
                in_table = True
                continue

            msg_match = self.msg_pattern.match(line)
            if msg_match:
                in_table = False # End table processing if a message appears
                current_msg_type = msg_match.group(1)
                content = msg_match.group(2)
                output["messages"].append({
                    "type": current_msg_type,
                    "content": content
                })
                continue
            
            if current_msg_type and not stripped.startswith("Section") and len(line) - len(line.lstrip()) > 4:
                if stripped:
                    output["messages"][-1]["content"] += " " + stripped
                continue

            if in_table and stripped:
                # Split by multiple spaces
                parts = self.table_row_split_pattern.split(stripped)
                metric_obj = {}
                
                if len(parts) >= 3:
                    metric_obj = {
                        "name": parts[0],
                        "unit": parts[1],
                        "value": self.parse_value(parts[2])
                    }
                elif len(parts) == 2:
                    metric_obj = {
                        "name": parts[0],
                        "unit": None,
                        "value": self.parse_value(parts[1])
                    }
                
                if metric_obj:
                    if current_section:
                        output["sections"][current_section].append(metric_obj)

        return output

