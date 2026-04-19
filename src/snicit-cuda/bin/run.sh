#!/bin/bash
log=tab4_fig11.txt

printf "Log File - " > $log
date >> $log

echo "Running SNICIT on benchmark A..."
info=`./main -k A | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on A ==, $info >> $log
# ////////////////////////////////////////////////////////////////


echo "Running SNICIT on benchmark B..."
info=`./main -k B | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on B ==, $info >> $log
# ////////////////////////////////////////////////////////////////////


echo "Running SNICIT on benchmark C..."
info=`./main -k C | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on C ==, $info >> $log
# ///////////////////////////////////////////////////////////////////


echo "Running SNICIT on benchmark D..."
info=`./main -k D | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on D ==, $info >> $log

cat $log
