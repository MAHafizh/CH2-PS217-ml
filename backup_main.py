@app.route('/get-image', methods=['GET'])
@token_required
def get_image(uid):
    try:
        filename = request.args.get('filename')
        
        if not uid:
            return jsonify({
                "status": {"code": 401, "message": "Unauthorized"},
                "error": "Invalid token or missing authentication"
        }), 401

        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f"userimages/{uid}/clothes/")


        get_images = []
        for i, blob in enumerate(blobs):
            filesname = blob.name
            file_name_ext = filesname.split("/")[-1]
            file_name_parts = file_name_ext.split(".")
            file_name = file_name_parts[0]

            metadata = blob.metadata
            result_category = metadata.get('category')
            result_color = metadata.get('color')

            image_folder = os.path.join("output_folder")
            os.makedirs(image_folder, exist_ok=True)

            url = f"https://storage.googleapis.com/{bucket.name}/{blob.name}"
            file_name_without_extension, file_extension = os.path.splitext(file_name)
            image_output = os.path.join(image_folder, f"{file_name_without_extension}.png")
            download_image(url, image_output)   

            selected_image = None

            if file_name == filename:
                selected_image = os.path.join("output_folder", f"{file_name}.png")
                image_data = {
                    "name": file_name,
                    "url": url,
                    "color": result_color,
                    "category": result_category,
                    "file_path": selected_image
                }
                get_images.append(image_data)  

            if selected_image is not None:
                print("selected img:", selected_image)
                return selected_image

        if not get_images:
            return jsonify({
                "status": {"code": 404, "message": "No Images Found with Specified Category"}
            }), 404

        return jsonify({
            "status": {"code": 200, "message": "Successfully Retrieved The Images"},
            "data": get_images
        }), 200  

    except Exception as error:
        return jsonify({
            "status": {"code": 500, "message": "Internal Server Error"},
            "data": {"error": str(error)}
        }), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


if selected_image:

            outfit_folder = create_outfits_folder(uid)

            for filename in os.listdir(local_folder_path):
                if filename.endswith(".png"):  # Ubah sesuai ekstensi gambar Anda
                    local_file_path = os.path.join("output_folder", filename)
                    remote_file_name = f"{outfit_folder}/{filename}" 

                    bucket = storage.bucket()
                    blob = bucket.blob(remote_file_name)
                    blob.upload_from_filename(local_file_path)

                    print(f"File {filename} uploaded to {remote_file_name}")

            return jsonify({
                "status": {"code": 200, "message": "Successfully Retrieved The Selected Image"},
                "data": {"selected_image": selected_image}
            }), 200
        else:
            return jsonify({
                "status": {"code": 404, "message": "No Images Found with Specified Category"}
            }), 404    