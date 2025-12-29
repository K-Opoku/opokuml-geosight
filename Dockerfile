FROM public.ecr.aws/lambda/python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN uv export --format requirements-txt --no-hashes > requirements.txt && \
    pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

COPY eurosat.onnx ${LAMBDA_TASK_ROOT}

COPY eurosat.onnx.data ${LAMBDA_TASK_ROOT}/eurosat.onnx.data

CMD ["lambda_function.lambda_handler"]