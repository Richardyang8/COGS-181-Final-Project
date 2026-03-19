"""
Chinese Classical Poetry Dataset
- Downloads Tang Dynasty poems from chinese-poetry/chinese-poetry GitHub repo
- Processes JSON into plain text for CharRNN training
- Each poem formatted as: title + author + content + separator
"""

import os
import json
import urllib.request


def download_chinese_poetry(save_dir="data", n_files=10):
    """
    Download Tang Dynasty poetry JSON files from GitHub.
    Each file contains ~1000 poems.

    Args:
        save_dir: directory to save files
        n_files: number of files to download (0-57 available, each ~1000 poems)
                 10 files ≈ 10,000 poems, good balance of size and training time

    Returns:
        path to the generated plain text file
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "chinese_poetry.txt")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Size: {len(text)} chars")
        return output_path

    base_url = ("https://raw.githubusercontent.com/chinese-poetry/"
                "chinese-poetry/master/全唐诗/poet.tang.{}.json")

    all_poems = []
    for i in range(n_files):
        url = base_url.format(i * 1000)
        json_path = os.path.join(save_dir, f"poet.tang.{i * 1000}.json")

        print(f"Downloading file {i+1}/{n_files}: poet.tang.{i * 1000}.json ...")
        try:
            urllib.request.urlretrieve(url, json_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                poems = json.load(f)
            all_poems.extend(poems)
            print(f"  Got {len(poems)} poems")
            # Clean up JSON file
            os.remove(json_path)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if not all_poems:
        raise RuntimeError("Failed to download any poetry files. Check network connection.")

    # Convert to plain text
    print(f"\nProcessing {len(all_poems)} poems...")
    text_lines = []
    for poem in all_poems:
        title = poem.get('title', '')
        author = poem.get('author', '')
        paragraphs = poem.get('paragraphs', [])

        if not paragraphs:
            continue

        # Format: title\nauthor\n content lines \n separator
        text_lines.append(title)
        text_lines.append(author)
        for line in paragraphs:
            text_lines.append(line)
        text_lines.append("")  # blank line separator

    full_text = "\n".join(text_lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"Saved: {output_path}")
    print(f"Total: {len(all_poems)} poems, {len(full_text)} characters")
    print(f"Unique characters: {len(set(full_text))}")
    print(f"Sample:\n{full_text[:300]}")

    return output_path


def create_compact_poetry(save_dir="data", n_files=5):
    """
    Create a smaller, content-only version (just poem lines, no titles/authors).
    Better for learning poetic patterns.
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "chinese_poetry_compact.txt")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return output_path

    base_url = ("https://raw.githubusercontent.com/chinese-poetry/"
                "chinese-poetry/master/全唐诗/poet.tang.{}.json")

    all_lines = []
    for i in range(n_files):
        url = base_url.format(i * 1000)
        json_path = os.path.join(save_dir, f"temp_tang_{i}.json")

        print(f"Downloading {i+1}/{n_files}...")
        try:
            urllib.request.urlretrieve(url, json_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                poems = json.load(f)
            for poem in poems:
                paragraphs = poem.get('paragraphs', [])
                for line in paragraphs:
                    # Clean: remove empty lines
                    line = line.strip()
                    if line:
                        all_lines.append(line)
            os.remove(json_path)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    full_text = "\n".join(all_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"Saved: {output_path} ({len(full_text)} chars, {len(all_lines)} lines)")
    return output_path


if __name__ == '__main__':
    path = download_chinese_poetry(n_files=10)
    print(f"\nDataset ready: {path}")
