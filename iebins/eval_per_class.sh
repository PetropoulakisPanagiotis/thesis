for i in {1..4}; do
    echo "Running for $i"
    python3 eval.py ../configs/arguments_eval_nyu_seg.txt $i
    if [ $i -ne 4 ]; then
        sleep 10  # Wait for 10 seconds before the next iteration
    fi
done
