// https://umijs.org/config/
import { defineConfig } from 'umi';
import defaultSettings from './defaultSettings';
import proxy from './proxy';
const { REACT_APP_ENV } = process.env;
export default defineConfig({
  hash: true,
  antd: {},
  dva: {
    hmr: true,
  },
  layout: {
    // https://umijs.org/zh-CN/plugins/plugin-layout
    locale: true,
    siderWidth: 220,
    ...defaultSettings,
  },
  // https://umijs.org/zh-CN/plugins/plugin-locale
  locale: {
    // enable: true,
    default: 'vi-VN',
    antd: true,
    // default true, when it is true, will use `navigator.language` overwrite default
    baseNavigator: false,
    // baseSeparator: '_',
  },
  dynamicImport: {
    loading: '@ant-design/pro-layout/es/PageLoading',
  },
  targets: {
    ie: 11,
  },
  // umi routes: https://umijs.org/docs/routing
  routes: [
    {
      path: '/admin/login',
      layout: false,
      hideInMenu: true,
      name: 'login',
      component: './user/Login/adminlogin',
    },
    {
      path: '/user',
      layout: false,
      routes: [
        {
          path: '/user/login',
          layout: false,
          name: 'login',
          // component: './user/Login',
          component: './auth',
        },

        {
          path: '/user',
          redirect: '/user/login',
        },
        {
          name: 'register',
          icon: 'smile',
          path: '/user/register',
          component: './user/register',
        },
        {
          component: '404',
        },
      ],
    },
    {
      hideInMenu: true,
      name: 'account',
      icon: 'user',
      path: '/account',
      routes: [
        {
          name: 'center',
          icon: 'smile',
          path: '/account/center',
          component: './account/center',
        },
      ],
    },
    {
      path: '/topic',
      name: 'danhsachtopic',
      // maChucNang: 'MENU',
      icon: 'BookOutlined',
      component: './QuanLyOutline',
      maChucNang: 'QUAN_LI_Outline',
    },
    {
      path: '/topic/:id/intent',
      name: 'danhsachintent',
      // maChucNang: 'MENU',
      icon: 'FieldStringOutlined',
      component: './QuanLyIntent',
      maChucNang: 'QUAN_LI_Intent',
      // access: 'CHU_DE',
      hideInMenu: true,
    },
    {
      path: '/topic/:id/intent/:id',
      name: 'chitietintent',
      // maChucNang: 'MENU',
      icon: 'UnorderedListOutlined',
      component: './QuanLyIntent/ChiTietIntent.tsx',
      hideInMenu: true,
    },
    {
      path: '/model',
      name: 'danhsachmodel',
      // maChucNang: 'MENU',
      icon: 'ShareAltOutlined',
      component: './QuanLyModel',
      // maChucNang: 'QUAN_LI_MODEL',
      // access: 'MO_HINH',
    },
    {
      path: '/fqas',
      name: 'fqas',
      // maChucNang: 'MENU',
      icon: 'MessageOutlined',
      component: './QuanLyCauHoiThuongGap',
      // maChucNang: 'QUAN_LI_MODEL',
      // access: 'MO_HINH',
    },
    // {
    //   path: '/users',
    //   name: 'quanlinguoidung',
    //   maChucNang: 'QUAN_LY_NGUOI_DUNG',
    //   // access: 'QUAN_LI_NGUOI_DUNG',
    //   icon: 'UnorderedListOutlined',
    //   routes: [
    //     {
    //       path: './danh-sach-nguoi-dung',
    //       name: 'danhsachnguoidung',
    //       icon: 'UnorderedListOutlined',
    //       // maChucNang: 'DANH_SACH_NGUOI_DUNG',
    //       component: './QuanLyNguoiDung',
    //       // access: 'NGUOI_DUNG',
    //     },
    //     {
    //       path: './danh-sach-nhom-nguoi-dung',
    //       name: 'danhsachnhomnguoidung',
    //       icon: 'UnorderedListOutlined',
    //       // maChucNang: 'DANH_SACH_NHOM_NGUOI_DUNG',
    //       component: './QuanLyNhomNguoiDung',
    //       // access: 'NHOM_NGUOI_DUNG',
    //     },
    //     {
    //       path: './bang-menu',
    //       name: 'Menu',
    //       // maChucNang: 'MENU',
    //       icon: 'UnorderedListOutlined',
    //       component: './QuanLyMenu',
    //       // access: 'MENU',
    //     },

    //   ],
    // },
    {
      layout: false,
      path: '/kichhoattaikhoan',
      component: './KichHoatTaiKhoan',
      hideInMenu: true,
      // access: 'thiSinhChuaKichHoat',
    },
    {
      layout: false,
      path: '/verifycccd',
      component: './VerifyCCCD',
      hideInMenu: true,
      // access: 'thiSinhChuaKichHoat',
    },

    {
      path: '/',
      redirect: '/user/login',
    },
    {
      component: '404',
    },
  ],
  // Theme for antd: https://ant.design/docs/react/customize-theme-cn
  theme: {
    'primary-color': defaultSettings.primaryColor,
  },
  // esbuild is father build tools
  // https://umijs.org/plugins/plugin-esbuild
  esbuild: {},
  title: false,
  ignoreMomentLocale: true,
  proxy: proxy[REACT_APP_ENV || 'dev'],
  manifest: {
    basePath: '/',
  },
  // Fast Refresh 热更新
  fastRefresh: {},

  nodeModulesTransform: {
    type: 'none',
  },
  // mfsu: {},
  webpack5: {},
  exportStatic: {},
});
