
#!/bin/bash
## step2 for pretrain the target steganalyzer
python3 step2.py -g 0 -p 0.4 -s hill -ln 1

## step3 for generate corresponding gradients maps
python3 step3.py -g 0 -p 0.4 -s hill -ln 1

