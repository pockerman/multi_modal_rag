import replicate



prompt="Fiberglass ship hull with severe buckling, large panel collapse and distorted plating with visible structural stress. Buckling is the structural deformation of the hull plating, where panels bend, bulge, or collapse under stress, typically due to compressive forces or impact. Realistic marine photography, inspection of vessel in dry dock, major damage clearly visible."

input={
            "prompt": prompt,
            "width": 768,
            "height": 512,
            "num_outputs": 2,  # generate 2 variations per severity
            "scheduler": "K_EULER",
            "guidance_scale": 7.5,
            "num_inference_steps": 50
}

output = replicate.run(
    #"stability-ai/sdxl::7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    input=input
)

# To access the file URLs:
print(output)
#=> "https://replicate.delivery/.../output_0.png"

# To write the files to disk:
for index, item in enumerate(output):
    with open(f"output_{index}.png", "wb") as file:
        file.write(item.read())
#=> output_0.png written to disk