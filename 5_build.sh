docker run  -p 9870:8971 -p 9873:9874 \
            --mount type=bind,source=/search/odin/lichangsong/zhengzhimigan/test/pb_process/success/,target=/models/test_model \
            -t tensorflow/serving:1.14.0   \
            --port=9871 --rest_api_port=9874 \
            --model_name=test_model --model_base_path=/models/test_model

