def p2c(py_b):
    str_b = str(py_b)
    c_b = str_b.replace("^", "**")
    return c_b

print(p2c('-0.164804744113+0.107478509525*((x-150)/35)-0.10544250283*((y-40)/35)+0.315356856341*((x-150)/35)^2+0.162472135933*((x-150)/35)^3-0.639801333627*((x-150)/35)^4-0.334724976583*((x-150)/35)^5-0.489726577083*((x-150)/35)*((y-40)/35)-0.615482007843*((x-150)/35)^2*((y-40)/35)+0.036197780289*((x-150)/35)^3*((y-40)/35)+0.6662313844*((x-150)/35)^4*((y-40)/35)+0.537731411089*((x-150)/35)^5*((y-40)/35)+1.1086155283*((y-40)/35)^2+0.337894874898*((y-40)/35)^3-0.379778527356*((y-40)/35)^4+0.021847467964*((x-150)/35)*((y-40)/35)^2+1.01805334847*((x-150)/35)*((y-40)/35)^3-0.257401018895*((x-150)/35)*((y-40)/35)^4-0.266226418542*((y-40)/35)^5-0.0692817978617*((x-150)/35)^2*((y-40)/35)^2+0.616060834279*((x-150)/35)^2*((y-40)/35)^3+0.307705648212*((x-150)/35)^2*((y-40)/35)^4-0.153011763442*((x-150)/35)^3*((y-40)/35)^2-0.344488232012*((x-150)/35)^3*((y-40)/35)^3-0.830757179123*((x-150)/35)^4*((y-40)/35)^2-0.299109873499*((x-150)/35)*((y-40)/35)^5+0.807113178005*((x-150)/35)^6-0.0331804670883*((y-40)/35)^6'))