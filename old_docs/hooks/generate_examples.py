import os
import shutil
import ast
import yaml
from pathlib import Path
from datetime import datetime

# In a MkDocs hook, paths are relative to the root folder (where mkdocs.yml lives)
ROOT_DIR = Path(".")
EXAMPLES_SRC = ROOT_DIR / "docs" / "examples"
GALLERY_DIR = ROOT_DIR / "docs" / "gallery"
GALLERY_POSTS = GALLERY_DIR / "posts"
ASSETS_DIR = ROOT_DIR / "docs" / "assets" / "examples"

def parse_metadata(file_path):
    with open(file_path, 'r') as f:
        try:
            tree = ast.parse(f.read())
            docstring = ast.get_docstring(tree)
            if docstring:
                metadata = yaml.safe_load(docstring)
                if isinstance(metadata, dict):
                    return metadata
        except (SyntaxError, yaml.YAMLError):
            pass
    return {}

def on_config(config, **kwargs):
    """
    This function is called by MkDocs at the start of the build process.
    We use it to generate the gallery and posts dynamically.
    """
    print(">>> Running Example Gallery Generation Hook...")
    
    # Ensure directories exist
    GALLERY_POSTS.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old posts
    for f in GALLERY_POSTS.glob("*.md"):
        f.unlink()

    examples_data = []
    
    # Iterate over folders in docs/examples/
    for example_dir in EXAMPLES_SRC.iterdir():
        if not example_dir.is_dir():
            continue
            
        py_files = list(example_dir.glob("*.py"))
        if not py_files:
            continue
            
        py_file = py_files[0]
        metadata = parse_metadata(py_file)
        
        if not metadata:
            print(f"Skipping {example_dir.name}: No metadata found.")
            continue
            
        title = metadata.get("Title", example_dir.name)
        tags = metadata.get("Tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
            
        image_name = metadata.get("Image")
        description = metadata.get("Description", "")
        
        # Prepare asset copy
        image_url = ""
        if image_name:
            src_image = example_dir / image_name
            if not src_image.exists():
                build_image = ROOT_DIR / "build" / image_name
                if build_image.exists():
                    src_image = build_image
            
            if src_image.exists():
                dest_asset_dir = ASSETS_DIR / example_dir.name
                dest_asset_dir.mkdir(exist_ok=True, parents=True)
                shutil.copy2(src_image, dest_asset_dir / image_name)
                # Path relative to docs/ for Zensical/MkDocs
                image_url = f"assets/examples/{example_dir.name}/{image_name}"
        
        # Post filename
        md_filename = f"{example_dir.name}.md"
        
        # Store for gallery index
        examples_data.append({
            "title": title,
            "description": description,
            "tags": tags,
            "image_url": image_url,
            "slug": example_dir.name,
            "md_path": f"posts/{md_filename}"
        })

        # Generate the Markdown post
        front_matter = {
            "title": title,
            "description": description,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "categories": tags,
            "tags": tags,
            "hide": ["navigation"]
        }
        if image_url:
            front_matter["image"] = "../../" + image_url

        with open(GALLERY_POSTS / md_filename, 'wt') as f:
            f.write("---\n")
            yaml.dump(front_matter, f, allow_unicode=True)
            f.write("---\n\n")
            
            f.write(f"{description}\n\n")
            
            if image_url:
                f.write(f"![{title}](../../{image_url}){{ align=center width=600 }}\n\n")
            
            f.write("## Python Implementation\n\n")
            f.write(f'--8<-- "{py_file}"\n')
            
    # Sort examples
    examples_data.sort(key=lambda x: x["title"])

    # Generate the Gallery Index with a GRID of CARDS
    with open(GALLERY_DIR / "index.md", 'wt') as f:
        f.write("# Examples\n\n")
        f.write('<div class="grid cards" markdown>\n\n')
        
        for ex in examples_data:
            f.write(f'::: [{ex["title"]}]({ex["md_path"]})\n')
            if ex["image_url"]:
                f.write(f'![{ex["title"]}](../{ex["image_url"]}){{ align=left width=150 }}\n\n')
            
            f.write(f'{ex["description"]}\n\n')
            
            if ex["tags"]:
                tags_str = ", ".join([f"`{t}`" for t in ex["tags"]])
                f.write(f"**Tags:** {tags_str}\n\n")
                
        f.write('</div>\n')

    print(f"Successfully generated {len(examples_data)} posts and gallery index.")
    return config
