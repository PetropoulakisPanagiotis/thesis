for i in {4..13}; do
    echo "Running for $i"
    python3 eval.py ../configs/arguments_eval_nyu.txt $i
    if [ $i -ne 13 ]; then
        sleep 10  # Wait for 10 seconds before the next iteration
    fi
done
