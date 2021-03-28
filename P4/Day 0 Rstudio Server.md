安装后，RStudio Server会自动启动运行。

```python
~ ps -aux|grep rstudio-server  # 查看RStudio Server运行进程
998       2914  0.0  0.1 192884  2568 ?        Ssl  10:40   0:00 /usr/lib/
rstudio-server/bin/rserver
```



#### 1.5.3　RStudio Server使用

通过浏览器，我们访问RStudio Server: 192.168.1.13:8787，IP地址为RStudio Server服务器的地址，如图1-6所示。



```R
sudo rstudio-server active-sessions
sudo rstudio-server stop
sudo rstudio-server status
sudo rstudio-server start
```



停止所有运行中的R进程：

```
sudo rstudio-server suspend-all
```



sudo firewall-cmd --zone=public --add-port=80/tcp --permanent 

firewall-cmd --list-ports