import React from 'react';
import { Button, Checkbox, Form, Input } from 'antd';
import { getInfo, login, write } from '@/services/Auth/auth';
import { Link, useModel } from 'umi';
import styles from './index.less';
import Title from 'antd/lib/typography/Title';
import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { history } from 'umi';
import data from '@/utils/data';

const Login: React.FC = () => {
  const { initialState, setInitialState } = useModel('@@initialState');
  const { loading, setLoading, loginModel } = useModel('auth');

  const onFinish = async (values: { username: string; password: string }) => {
    // console.log('Success:', values);
    const formdata = new FormData();
    formdata.append('username', values.username);
    formdata.append('password', values.password);
    
    try {
      const res = await loginModel(formdata);
      if (res.data?.error?.status === 403) {
        console.log("Forbidden");
  
        localStorage.removeItem("token");
      }
      if (res.status === 200 && res.data?.jwt) {
        localStorage.setItem('token', res.data?.jwt);
        try {
          await write(res.data?.jwt);
          const info = await getInfo();
          localStorage.setItem('username', info.data?.username);
          console.log('info', info);
          if (info.status === 200) {
            let systemRole = 'Admin';
            // if (info.data?.is_superuser === true) {
            //   systemRole = 'Admin';
            // } else if (info.data?.is_staff === true) {
            //   systemRole = 'Staff';
            // } else {
            //   systemRole = 'User';
            // }
            localStorage.setItem('vaiTro', systemRole);
            setInitialState({
              ...initialState,
              currentUser: {
                ...info.data,
                systemRole: systemRole,
              },
            });
            history.push(data?.path?.[systemRole] ?? '/');
            return;
          }
        }
        catch (err) {
          localStorage.removeItem("token");
          console.log(err);
        }
      }
      history.push('/');
    }
    catch (err) {
      localStorage.removeItem("token");
      console.log(err);
    }
    
    // console.log(res);

    
  };

  const onFinishFailed = (errorInfo: any) => {
    console.log('Failed:', errorInfo);
  };

  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <div className={styles.top}>
          <Link to="/" className={styles.header}>
            <img alt="logo" className={styles.logo} src="/logo.png" />
            <Title level={1} className={styles.title}>
              Chatbot Data server
            </Title>
          </Link>
        </div>

        <div className={styles.main}>
          <Form
            name="normal_login"
            className={styles.loginForm}
            initialValues={{ remember: true }}
            onFinish={onFinish}
          >
            <Form.Item name="username" rules={[{ required: true, message: 'Nhập tài khoản!' }]}>
              <Input
                prefix={<UserOutlined className="site-form-item-icon" />}
                placeholder="Tài khoản"
              />
            </Form.Item>
            <Form.Item name="password" rules={[{ required: true, message: 'Nhập mật khẩu!' }]}>
              <Input.Password
                prefix={<LockOutlined className="site-form-item-icon" />}
                type="password"
                placeholder="Mật khẩu"
              />
            </Form.Item>
            <Form.Item>
              <Form.Item name="remember" valuePropName="checked" noStyle>
                <Checkbox>Nhớ tài khoản</Checkbox>
              </Form.Item>

              <a className={styles.loginFormFogot} href="">
                Quên mật khẩu
              </a>
            </Form.Item>

            <Form.Item>
              <Button type="primary" htmlType="submit" className={styles.loginFormButton}>
                Đăng nhập
              </Button>
              <a href="" className={styles.loginFormRegister}>
                Đăng ký
              </a>
            </Form.Item>
          </Form>
        </div>
      </div>
    </div>
  );
};

export default Login;
