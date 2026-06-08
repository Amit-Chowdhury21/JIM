import glob
import re
files = glob.glob('dashboard/src/**/*.jsx', recursive=True) + glob.glob('dashboard/src/**/*.js', recursive=True)
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    # Remove the orphaned values for genetic and nlp in objects
    new_content = re.sub(r'/\*\s*genetic:\s*removed\s*\*/\s*\'[^\']+\'\s*,?', '', new_content)
    new_content = re.sub(r'/\*\s*nlp:\s*removed\s*\*/\s*\'[^\']+\'\s*,?', '', new_content)
    new_content = re.sub(r'/\*\s*genetic:\s*removed\s*\*/\s*\"[^\"]+\"\s*,?', '', new_content)
    new_content = re.sub(r'/\*\s*nlp:\s*removed\s*\*/\s*\"[^\"]+\"\s*,?', '', new_content)
    
    # Or for numbers
    new_content = re.sub(r'/\*\s*genetic:\s*removed\s*\*/\s*[\d\.]+\s*,?', '', new_content)
    new_content = re.sub(r'/\*\s*nlp:\s*removed\s*\*/\s*[\d\.]+\s*,?', '', new_content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed {file_path}')
