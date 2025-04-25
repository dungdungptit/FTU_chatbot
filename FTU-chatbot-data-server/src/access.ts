import type { Login } from './services/ant-design-pro/typings';
import { ESystemRole } from './utils/constants';
import { handlePhanNhom } from './utils/utils';

/**
 * @see https://umijs.org/zh-CN/plugins/plugin-access
 * */

const checkCode = (code: string, list: any[]) => {
  let result = false;
  list?.forEach((item) => {
    if (item?.code === code) {
      console.log('true');
      result = true;
    }
  });
  return result;
}

export default function access(initialState: {
  currentUser: Login.Profile;
  phanNhom: {
    userId: string;
    danhSachPhanNhom: {
      mucDo: string;
      nhomVaiTroId: {
        _id: string;
        danhSachChucNang: string[];
      };
    }[];
    vaiTro: string;
  };
}) {
  const vaiTro = initialState?.currentUser?.systemRole;
  const menulist = initialState?.currentUser?.menus;
  const verifiedEmail = initialState?.currentUser?.emailVerify?.verified ?? false;
  const verifiedCCCD = initialState?.currentUser?.cmtCccd !== undefined;
  const token = localStorage.getItem('token');
  console.log('menulist', menulist);
  console.log(token && checkCode('QUAN_LI_NGUOI_DUNG', menulist));
  return {
    QUAN_LI_NGUOI_DUNG: token && menulist && checkCode('QUAN_LI_NGUOI_DUNG', menulist),
    NGUOI_DUNG: token && menulist && checkCode('NGUOI_DUNG', menulist),
    NHOM_NGUOI_DUNG: token && menulist && checkCode('NHOM_NGUOI_DUNG', menulist),
    MENU: token && menulist && checkCode('MENU', menulist),
    CHU_DE: token && menulist && checkCode('CHU_DE', menulist),
    MO_HINH: token && menulist && checkCode('MO_HINH', menulist),
    thiSinh: token && verifiedEmail && verifiedCCCD && vaiTro === ESystemRole.ThiSinh,
    thiSinhChuaKichHoat: token && vaiTro === ESystemRole.ThiSinh,
    quanTriVien: token && vaiTro && vaiTro === ESystemRole.QuanTriVien,
    adminVaQuanTriVien:
      token && vaiTro && (vaiTro === ESystemRole.Admin || vaiTro === ESystemRole.QuanTriVien),
    routeFilter: (route: any) => {
      return handlePhanNhom(initialState, route?.maChucNang);
    },
  };
}
