// Driver manager for browser that selects WebNN -> WebGPU -> WASM -> JS fallback
import('./hw/auditor.js').then(({ auditHardware }) => {
  // no-op loader placeholder for bundlers; exports at runtime
});

// Provide a small manager to select best available driver
export async function createDriverManager() {
  const { auditHardware } = await import('./hw/auditor.js');
  const profile = await auditHardware();

  if (profile.hasWebNN) {
    const mod = await import('./hw/webnn_driver.js');
    const driver = new mod.WebNNDriver();
    await driver.init();
    return { driver, profile };
  }

  if (profile.hasWebGPU) {
    const mod = await import('./hw/webgpu_driver.js');
    const driver = new mod.WebGPUDriver();
    await driver.init();
    return { driver, profile };
  }

  // WASM or JS fallback
  try {
    const mod = await import('./hw/wasm_driver.js');
    const driver = new mod.WasmDriver();
    await driver.init();
    return { driver, profile };
  } catch (e) {
    const mod = await import('./hw/wasm_driver.js');
    const driver = new mod.WasmDriver();
    await driver.init();
    return { driver, profile };
  }
}
