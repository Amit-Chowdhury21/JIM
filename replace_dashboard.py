import os
import glob
import re

dashboard_dir = 'dashboard/src'
files = glob.glob(f'{dashboard_dir}/**/*.jsx', recursive=True) + glob.glob(f'{dashboard_dir}/**/*.js', recursive=True)

replacements = {
    'wavelet:': 'wavelet_pro:',
    'hmm:': 'hmm_pro:',
    'tft:': 'tft_pro:',
    "'wavelet'": "'wavelet_pro'",
    '"wavelet"': '"wavelet_pro"',
    "'hmm'": "'hmm_pro'",
    '"hmm"': '"hmm_pro"',
    "'tft'": "'tft_pro'",
    '"tft"': '"tft_pro"',
    'genetic:': '/* genetic: removed */',
    'nlp:': '/* nlp: removed */',
    "'genetic',": '',
    "'nlp',": '',
    'log.wavelet_signal': 'log.wavelet_pro_signal',
    'log.wavelet_conf': 'log.wavelet_pro_conf',
    'log.hmm_signal': 'log.hmm_pro_signal',
    'log.hmm_conf': 'log.hmm_pro_conf',
    'log.tft_signal': 'log.tft_pro_signal',
    'log.tft_conf': 'log.tft_pro_conf',
    'modelMetrics.wavelet': 'modelMetrics.wavelet_pro',
    'modelMetrics.hmm': 'modelMetrics.hmm_pro',
    'modelMetrics.tft': 'modelMetrics.tft_pro',
}

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    new_content = re.sub(r',\s*\'genetic\'\s*', '', new_content)
    new_content = re.sub(r',\s*\'nlp\'\s*', '', new_content)
    new_content = re.sub(r'\'genetic\'\s*,', '', new_content)
    new_content = re.sub(r'\'nlp\'\s*,', '', new_content)
    
    # fix object keys
    new_content = new_content.replace('wavelet_pro_pro:', 'wavelet_pro:')
    new_content = new_content.replace('hmm_pro_pro:', 'hmm_pro:')
    new_content = new_content.replace('tft_pro_pro:', 'tft_pro:')
    new_content = new_content.replace("'wavelet_pro_pro'", "'wavelet_pro'")
    new_content = new_content.replace("'hmm_pro_pro'", "'hmm_pro'")
    new_content = new_content.replace("'tft_pro_pro'", "'tft_pro'")
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {file_path}')
