function m = py_decode(z)
    global encoder
    mod = py.importlib.import_module('load_encoder');
    m = squeeze(double(mod.decode(encoder, z))); 
end
