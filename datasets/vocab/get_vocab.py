import os
from pathlib import Path

vecs = list(Path("/ceph/tgreen/projects/babelnet_extraction/vocab").glob("*vec*"))
print(vecs)
for x in vecs:
    x = x.name
    print(f"Processing {x}")
    lang = x.split(".")[1]
    f = open(f"/ceph/tgreen/projects/babelnet_extraction/vocab/{x}")
    words = [
        y.split(" ")[0] + "\n"
        for idx, y in enumerate(f.readlines())
        if 100000 >= idx > 0
    ]
    f_out = open(
        f"/ceph/tgreen/projects/babelnet_extraction/vocab/vocab_files/{lang}_vocab.txt",
        "w",
    )
    f_out.writelines(words)
    f_out.close()
    f.close()
