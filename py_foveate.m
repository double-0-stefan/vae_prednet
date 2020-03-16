function g = py_foveate(im, a)
  
    m = getEncoder();
    mod = py.importlib.import_module('load_encoder');
    g = squeeze(double(mod.foveate(m, im, a))); 
    
end
