import joblib


def produce_linear_prediction_c_code(model):
    thetas = f"float thetas[] = {{"
    for i in model.coef_:
        thetas += f"{float(i)}, "
    thetas += f"}};"

    bias = f"{model.intercept_};"
    code = f"""

        float prediction(float *features, int n_features) {{

            {thetas}

            float res = {bias}
            for (int i = 0; i < n_features; ++i) {{
                res += features[i] * thetas[i];
            }}

            return res;
        }}

    """
    return code

def produce_main(features):
    arr = f"float arr[] = {{"
    for i in features:
        arr += f"{i}, "
    arr += f"}};"

    code = f"""
        #include <stdio.h>
        #include <stdlib.h>
        
        int main() {{
            {arr}
            float *arr_heap = malloc (sizeof (float) * {len(features)});

            for (int i = 0; i < {len(features)}; ++i) {{
                arr_heap[i] = arr[i];
            }} 
            
            printf("Linear regression: %f\\n", prediction(arr_heap, {len(features)}));
            free(arr_heap);

            return 0;
        }} 
    
    """
    
    return code

def transpile_model(features):
    model = joblib.load("regression.joblib")

    c_main = produce_main(features)
    c_code = produce_linear_prediction_c_code(model)

    f = open('transpiler.c', 'w')
    f.write(c_code + c_main)
    f.close()