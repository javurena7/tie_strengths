#!/bin/bash
# Used for obtaning extended_net.edg from /m/cs/scratch/networks-mobile/set5/original
awk '{($5 > $3) ? p = $3 FS $5 : p = $5 FS $3; print p}' $1 | sort | uniq -c | awk '{print $2, $3, $1}' > $2
