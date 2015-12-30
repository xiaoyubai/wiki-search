# bash command to recover a reset hard
# create a hashfile
# recover from hashfile
# git fsck --cache --unreachable $(git for-each-ref --format="%(wiki-search)") > allhashes


commits=["e15a90e145918419611fbfcbe793a8d419320b06",
"4b825dc642cb6eb9a060e54bf8d69288fbee4904",
"0d575e6eb7148b69eb3b8216ddb321ae7173d050",
"af7fb276b6778ea6be1e052f8ee2b1d8d18504cd",
"4f6a3436304a838e398f79fb4affafe3082b487d",
"715b62ac2eaeef43fe97951b5d92f227442cefa7",
"778a154cc54337352c8e235ad08f355203df9e7a",
"5e39ab103a5207fee3a298d90a2a1f1541668919"]

from subprocess import call
filename = "file"
i = 1
for c in commits:
    f = open(filename + str(i), "wb")
    call(["git", "show", c], stdout=f)
    i+=1
