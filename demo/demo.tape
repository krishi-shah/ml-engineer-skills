# Renders the README demo GIF with VHS (https://github.com/charmbracelet/vhs).
#   1. install VHS:            https://github.com/charmbracelet/vhs#installation
#   2. from the repo root run: vhs demo/demo.tape
#   3. commit docs/demo.gif and uncomment the image line in README.md
Output docs/demo.gif

Set FontSize 18
Set Width 1200
Set Height 760
Set Padding 20
Set Theme "Dracula"

Type "mlcheck scan-source example/leaky_pipeline.py --suggest"
Sleep 800ms
Enter
Sleep 4s

Type "mlcheck audit --source example/leaky_pipeline.py --repro"
Sleep 800ms
Enter
Sleep 4s
