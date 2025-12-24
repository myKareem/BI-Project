import os
import subprocess
import glob
import sys  # <--- Make sure this is imported

def run_all_hypergraphs():
    folder_path = "Hypergraphs"
    files = []
    
    # Extensions to look for
    extensions = ['*.dtl', '*.hg', '*.gml'] 
    
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print("No files found in Hypergraphs/ folder!")
        return

    print(f"Found {len(files)} hypergraphs. Starting processing...\n")

    for full_path in files:
        filename = os.path.basename(full_path)
        
        print(f"------------------------------------------------")
        print(f" Running Ralph on: {filename}")
        print(f"------------------------------------------------")
        
        try:
            # FIX: Use sys.executable to ensure we use the 'cplex_env' python
            subprocess.run([sys.executable, "fhtw.py", filename], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f" Error running {filename}: {e}")
        except Exception as e:
            print(f" Unexpected error: {e}")

if __name__ == "__main__":
    run_all_hypergraphs()