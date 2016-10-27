# prints the validation accuracies in a python list format
BEGIN {i=1; printf "\"[";}
/validation accuracy/ {printf "(%d, %s),", i, $3; i++}
END {printf "]\"";}
