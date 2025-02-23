from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import base64

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)

            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)

            try:  # 予測処理でエラーが発生した場合の処理
                preds = model.predict(img_array)
                top_preds = decode_predictions(preds, top=5)[0]

                predictions = [] #追加
                for pred in top_preds:
                    predictions.append({
                        'class': pred[1],
                        'probability': pred[2] * 100
                    })

                encoded_img = base64.b64encode(img_file.getvalue()).decode('utf-8')
                img_data = f"data:image/jpeg;base64,{encoded_img}"

                return render(request, 'home.html', {
                    'form': form,
                    'predictions': predictions, # predictionsを渡す
                    'img_data': img_data
                })
            except Exception as e:
                print(f"Prediction Error: {e}")
                return render(request, 'home.html', {'form': form, 'error_message': '予測処理でエラーが発生しました。'})

        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})