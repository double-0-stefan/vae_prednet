function z = py_encode(a)
    m = getEncoder();
    s = getStimulus();
    z = m.model(full(s), pyargs('actions', a, 'to_matlab', true));
    z = double(z{1}.single);
end