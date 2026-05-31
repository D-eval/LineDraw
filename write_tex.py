import re

table_file = "./table.txt"
tex_file = "./paper/a.tex"

tag = "linedraw_threshold"

# 读取表格
with open(table_file, "r", encoding="utf-8") as f:
    table_content = f.read().rstrip()

# 读取 tex
with open(tex_file, "r", encoding="utf-8") as f:
    tex = f.read()

start_tag = f"%wssb:start:{tag}"
end_tag = f"%wssb:end:{tag}"

pattern = (
    re.escape(start_tag)
    + r".*?"
    + re.escape(end_tag)
)

replacement = (
    start_tag
    + "\n"
    + table_content
    + "\n"
    + end_tag
)

new_tex, n = re.subn(
    pattern,
    lambda m: replacement,
    tex,
    flags=re.DOTALL
)

if n == 0:
    raise RuntimeError(
        f"Tag not found: {tag}"
    )

with open(tex_file, "w", encoding="utf-8") as f:
    f.write(new_tex)

print(
    f"Updated {tex_file} "
    f"({tag})"
)