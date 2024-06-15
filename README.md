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

1. ```get_noise_interp_multi()``` (DONE)
2. ```generate_waveform()```

For ```python generate_waveform()``` we have to implement the following methods:

```python
distribute_gen(...)
distribute_dec2(...)
distribute_dec(...)
conc_tog_specphase(...)
```

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

### Test to generate stereo audio
easy test would be to write ```U.generate_example_stereo()```.

Here we have to implement the following methods into tensorflowjs:

```python
distribute_dec2(...)
distribute_dec(...)
conc_tog_specphase(...)
denormalize(...)
db2power(...)
```