# MusikaWeb

## Resources

* Colab notebook to re-load and convert models 

## TODO List

### Convert all models (DONE)

Convert all necessary models from ```get_networks()``` as SavedModel.

```python
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch
= M.get_networks()
dec.save('models/dec')
dec2.save('models/dec2')
gen_ema.save('models/gen_ema')
```

```bash
!for path in models/*; do tensorflowjs_converter
--input-format=tf_saved_model $path web$path; done
!zip -r web_models.zip web_models
```

### Run all models (DONE)

Try to run all three models.

### Write generation functions in typescript

```get_noise_interp_multi()``` and ```generate_waveform()```
https://chatgpt.com/share/9612ed76-1137-492f-8a35-63a977ecd9d8

The way to go here would be to rewrite every function and also write tests.

other option:

```python
critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch
= M.get_networks()
inpf = tf.keras.layers.Input((M.args.latlen, M.args.latdepth * 2))
allinone = tf.keras.Model(inpf, [U.generate_waveform(inpf, gen_ema, dec,
dec2, batch_size=64)])
allinone.save('models/allinone')
```

easy test would be to write ```U.generate_example_stereo()```.