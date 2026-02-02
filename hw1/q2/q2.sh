#!/bin/bash

function timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }

gspan_path="$1"
fsg_path="$2"
gaston_path="$3"
input_path=$4
output_path=$5
convert_path="converttoformat.py"

gspan_path=$(realpath "$gspan_path")
fsg_path=$(realpath "$fsg_path")
gaston_path=$(realpath "$gaston_path")
input_path=$(realpath "$input_path")
output_path=$(realpath "$output_path")
convert_path=$(realpath "$convert_path")
#=================================
log_path=$output_path/run_log.txt
mkdir -p "$output_path"
#delete log file if exist
if [ -f "$log_path" ]; then
    echo "File exists. Deleting..."
    rm "$log_path"
fi

# Create a new file
touch "$log_path"
echo "New file created: $log_path"
#=================================



if [ -f $convert_path ]; then
    echo "Converting dataset format..."
    python3 converttoformat.py "$input_path"
else
    echo "Error: converttoformat.py not found!"
    exit 1
fi


support_list=(95 50 25 10 5)


total_graph=$(grep -c "^#" "$input_path")
echo "Total number of graphs: $total_graph"

for support in "${support_list[@]}"; do

    support_perc=$(echo "scale=4; $support / 100" | bc)

    gspan_outfile="$output_path/gspan$support"
    fsg_outfile="$output_path/fsg$support"
    gaston_outfile="$output_path/gaston$support"

    touch $gspan_outfile
    touch $fsg_outfile
    touch $gaston_outfile

    echo "Running gSpan for support $support%"
    start_time=$(date +%s)
    echo "Executing: timeout 3600 $gspan_path -s $support_perc -f gspan.txt -o"
    timeout 3600 $gspan_path -s $support_perc -f gspan.txt -o
    status=$? || true
    end_time=$(date +%s)
    gspan_totaltime=$((end_time - start_time))
    echo "gSpan runtime at $support% support: $gspan_totaltime seconds" >> "$log_path"
    if [ $status -eq 124 ]; then
        echo "gSpan timeout 1hr" | tee -a "$log_path"
        > "$fsg_outfile"
    elif [ $status -eq 139 ]; then
        echo "gSpan crashed with segmentation fault (SIGSEGV)" | tee -a "$log_path"
        > "$gspan_outfile"
    elif [ $status -eq 137 ]; then
        echo "gSpan was killed (SIGKILL)" | tee -a "$log_path"
        > "$gspan_outfile"
    elif [ $status -eq 143 ]; then
        echo "gSpan was terminated (SIGTERM)" | tee -a "$log_path"
        > "$gspan_outfile"
    else
        
        if [ -f "gspan.txt.fp" ]; then
            cp "gspan.txt.fp" "$gspan_outfile"
        fi
    fi

    echo "Running FSG for support $support%"
    start_time=$(date +%s)
    echo "Executing: timeout 3600 $fsg_path -s $support fsg.txt $fsg_outfile"
    timeout 3600 $fsg_path -s $support fsg.txt $fsg_outfile
    status=$? || true
    end_time=$(date +%s)
    fsg_totaltime=$((end_time - start_time))
    
    echo "FSG runtime at $support% support: $fsg_totaltime seconds" >> "$log_path"
    if [ $status -eq 124 ]; then
        echo "FSG timeout 1hr" | tee -a "$log_path"
        > "$fsg_outfile"
    elif [ $status -eq 139 ]; then
        echo "FSG crashed with segmentation fault (SIGSEGV)" | tee -a "$log_path"
        > "$fsg_outfile"
    elif [ $status -eq 137 ]; then
        echo "FSG was killed (SIGKILL)" | tee -a "$log_path"
        > "$fsg_outfile"
    elif [ $status -eq 143 ]; then
        echo "FSG was terminated (SIGTERM)" | tee -a "$log_path"
        > "$fsg_outfile"
    else
        
        if [ -f "fsg.fp" ]; then
            cp "fsg.fp" "$fsg_outfile"
        fi

    fi

    gaston_support=$(echo "$total_graph * $support_perc" | bc -l | awk '{printf "%.0f\n", $1}')
    echo "Running Gaston for support $support%"
    start_time=$(date +%s)
    echo "Executing: timeout 3600 $gaston_path $gaston_support gaston.txt $gaston_outfile"
    timeout 3600 $gaston_path $gaston_support gaston.txt $gaston_outfile
    status=$?
    end_time=$(date +%s)
    gaston_totaltime=$((end_time - start_time))
    
    echo "Gaston runtime at $support% support: $gaston_totaltime seconds" >> "$log_path"
    if [ $status -eq 124 ]; then
        echo "Gaston timeout 1hr" | tee -a "$log_path"
        > "$gaston_totaltime"
    elif [ $status -eq 139 ]; then
        echo "Gaston crashed with segmentation fault (SIGSEGV)" | tee -a "$log_path"
        > "$gaston_totaltime"
    elif [ $status -eq 137 ]; then
        echo "Gaston was killed (SIGKILL)" | tee -a "$log_path"
        > "$gaston_totaltime"
    elif [ $status -eq 143 ]; then
        echo "Gaston was terminated (SIGTERM)" | tee -a "$log_path"
        > "$gaston_totaltime"
    fi

done

python3 q2.py "$output_path"
