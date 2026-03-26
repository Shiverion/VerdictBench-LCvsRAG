import sys

file_path = r"c:\Users\miqba\projects\LC vs RAG benchmark\experiments\pipeline\runner.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_loop = False

for i, line in enumerate(lines):
    if "for i, pair in enumerate(tqdm(qa_pairs" in line:
        in_loop = True
        new_lines.append(line)
        continue
        
    if in_loop:
        if line.startswith("            cost_tracker.add(CostRecord("):
            new_lines.append("    " + line)
            continue
        
        # If it's returning to unindented stuff at the end (like Save cost summaries)
        if line.startswith("    # Save cost summaries"):
            in_loop = False
            # Insert our missing check and except block first
            new_lines.append("                breaker.check(gen_meter, judge_meter, label=f\"{system.condition_name}[{i+1}]\")\n")
            new_lines.append("\n        except BudgetExceeded as e:\n")
            new_lines.append("            log.error(f\"Run stopped early! Partial results preserved. Exception: {e}\")\n\n")
            new_lines.append(line)
            continue
            
        # Re-indent the loop body by 4 spaces
        if line.strip() == "":
            new_lines.append("\n")
        else:
            if line.startswith("            "):
                new_lines.append("    " + line)
            elif line.startswith("        "):
                new_lines.append("    " + line)
            else:
                new_lines.append(line)
    else:
        new_lines.append(line)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Formatting applied.")
