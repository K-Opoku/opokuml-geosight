import lambda_function
with open("test_image.jpg", "rb") as f:
    image_bytes = f.read()

fake_event = {
    "image_bytes": image_bytes
}

# 3. calling function for prediction to be made
print("Sending to model...")
result = lambda_function.lambda_handler(fake_event, None)

print("RESULT:")
print(result)