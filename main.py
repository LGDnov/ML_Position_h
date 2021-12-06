import function

#test run for the torch model
model = function.loading('model\model_my01.pth')
function.predict_m("test","side",model)


