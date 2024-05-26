SAVE_PATH=$1
MATHPILE_PATH=$2
mkdir -p $SAVE_PATH

diff=mix
python -m synthesis_cot.build_math_stack_prompts \
    --save_path $SAVE_PATH/stack \
    --prompt_path prompts/cot \
    --generation_style $diff \
    --num_chunks 1
    

diff=mix
python -m synthesis_cot.build_math_textbook_prompts \
    --data_dir $MATHPILE_PATH \
    --save_path $SAVE_PATH/textbook \
    --prompt_path prompts/cot \
    --generation_style $diff \
    --num_chunks 1 \

diff=mix 
python -m synthesis_cot.build_math_wiki_prompts \
    --data_dir $MATHPILE_PATH \
    --save_path $SAVE_PATH/wiki \
    --prompt_path prompts/cot \
    --generation_style $diff \
    --num_chunks 1
    
diff="mix"
python -m synthesis_cot.build_math_web_prompts \
    --save_path $SAVE_PATH/web \
    --prompt_path prompts/cot \
    --generation_style $diff \
    --num_chunks 1  \
    --scope "0.20-1.00"

diff="mix"
python -m synthesis_cot.build_math_arxiv_prompts \
    --save_path $SAVE_PATH/arxiv \
    --prompt_path prompts/cot \
    --generation_style $diff \
    --num_chunks 1 \
    --scope "0.60-1.00"

python -m synthesis_cot.collect_source_data \
    --data_path $SAVE_PATH