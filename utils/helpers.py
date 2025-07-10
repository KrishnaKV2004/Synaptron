def print_model_summary(model):

    print("Model Summary :")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.__class__.__name__} | units: {getattr(layer, 'units', 'NA')}")