black "."
pdoc --html tinydl
jupytext --to notebook "demo.py"
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
