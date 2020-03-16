function g = py_decode(z)
  
    m = getEncoder();
    mod = py.importlib.import_module('load_encoder');
    g = squeeze(double(mod.decode(m, z))); 
    
end
