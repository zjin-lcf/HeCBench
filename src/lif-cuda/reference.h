void reference (
    size_t numNeurons, int neurons_per_item, float dt,
    float*__restrict__ encode_result,
    float*__restrict__ voltage_array,
    float*__restrict__ reftime_array,
    float tau_rc, float tau_ref,
    float*__restrict__ bias,
    float*__restrict__ gain,
    float*__restrict__ spikes)
{
  for (size_t i = 0; i < numNeurons; i++)
  {
    int neuron_index = i % neurons_per_item;
    int item_index = i / neurons_per_item;

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -expm1f(-dt / tau_rc) * (current - voltage);
    voltage = fmaxf(voltage + dV, 0.f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.f / dt;
    mult += 1.f;

    mult = mult > 1.f ? 1.f : mult;
    mult = mult < 0.f ? 0.f : mult;

    voltage *= mult;

    if(voltage > 1.f){
      spike = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage = 0.f;
    }else{
      spike = 0.f;
    }

    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}


