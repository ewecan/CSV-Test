char var;

void setup()
{
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);    //设置13号端口作为输出端口
    digitalWrite(LED_BUILTIN, HIGH); //让灯开始时亮
}

void SendData()
{
    int a[3];
    a[0] = analogRead(A0);
    a[1] = analogRead(A1);
    a[2] = analogRead(A2);

    Serial.print(a[0]);
    Serial.print(" ");
    Serial.print(a[1]);
    Serial.print(" ");
    Serial.println(a[2]);
}

void RecvData()
{
    while (Serial.available() > 0)
    {
        var = Serial.read();
        // Serial.print(var);
        if (var == 'N')
            digitalWrite(LED_BUILTIN, LOW);
        if (var == 'D')
            digitalWrite(LED_BUILTIN, HIGH);
    }
}

void loop()
{
    SendData();
    RecvData();
}