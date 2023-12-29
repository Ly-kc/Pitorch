## 思路
框架名：pitorch
将value类中的numpy换成raw_pisor，再将value包装成pisor
operater forward方法输入输出改成raw_pisor   gradient的输入输出仍然为pisor

一视同仁，cpu和gpu都在raw_pisor层级上进行计算，不过需要确保raw_pisor已经实现了相应运算的前端。若对于某些功能实在懒得实现，可以先转成numpy，算完之后再转回来

若要搞梯度的梯度，需要将conv和pool的backward也用基本算子实现，或许可以先不管

废除标量，在转换为rawPisor前将其变为(1,)的array--------**有待考虑**