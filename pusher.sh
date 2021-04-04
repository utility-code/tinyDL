black "."
pdoc --o docs tinydl
mv docs/tinydl/* docs/
jupytext --to notebook "demo.py"
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
