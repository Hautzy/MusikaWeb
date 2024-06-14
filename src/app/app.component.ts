import {Component, OnInit} from '@angular/core';
import { RouterOutlet } from '@angular/router';
import * as tf from '@tensorflow/tfjs';
import {zip} from "rxjs";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit {
  async ngOnInit(): Promise<void> {
    const dec_model = await tf.loadGraphModel('./assets/models/dec_model_web/model.json');
    const dec2_model = await tf.loadGraphModel('./assets/models/dec2_model_web/model.json');
    const gen_ema_model = await tf.loadGraphModel('./assets/models/gen_ema_model_web/model.json');

    const model_names = ['dec_model', 'dec2_model', 'gen_ema_model'];
    const models = [dec_model, dec2_model, gen_ema_model];

    console.log('Models loaded');
    await zip(model_names, models).forEach(([name, model]) => {
      console.log(name);
      console.log('Inputs and outputs:');
      console.log(model.inputs);
      console.log(model.outputs);
    });
  }
}
