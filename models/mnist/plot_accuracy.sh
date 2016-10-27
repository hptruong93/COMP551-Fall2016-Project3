# redirect output of train_mnist.py to a file and run this script with the file as the arg to see a plot
# of validation curve as you go
awk -f get_accuracies.awk $1 | xargs python -c "import sys;import matplotlib.pyplot as plt;plt.plot(*zip(*eval(sys.argv[1])));plt.show();"
