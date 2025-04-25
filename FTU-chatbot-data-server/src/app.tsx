import RightContent from '@/components/RightContent';
import type { Settings as LayoutSettings } from '@ant-design/pro-layout';
import { PageLoading } from '@ant-design/pro-layout';
import { notification, Tooltip } from 'antd';
import 'moment/locale/vi';
import type { RequestConfig, RunTimeLayoutConfig } from 'umi';
import { getIntl, getLocale, history } from 'umi';
import type { RequestOptionsInit, ResponseError } from 'umi-request';
import ErrorBoundary from './components/ErrorBoundary';
import NotAccessible from './pages/exception/403';
import NotFoundContent from './pages/exception/404';
import { getInfo, getInfoAdmin } from './services/ant-design-pro/api';
import type { Login } from './services/ant-design-pro/typings';
import { ESystemRole } from './utils/constants';
import data from './utils/data';
// import { getPhanNhom } from './utils/utils';

const loginPath = '/user/login';
const pathAuth = ['/admin/login', '/user/quenMatKhau'];
/**  loading */
export const initialStateConfig = {
  loading: <PageLoading />,
};

/**
 * @see  https://umijs.org/zh-CN/plugins/plugin-initial-state
 * */
export async function getInitialState(): Promise<{
  settings?: Partial<LayoutSettings>;
  // currentUser?: Login.User
  currentUser?: {
    username: string;
    systemRole: string;
    menus: [];
  };
  partner_id?: number;
  fetchUserInfo?: () => Promise<{ data: { data: Login.User } } | undefined>;
  authorizedRoles?: any[];
  phanNhom?: {
    userId: string;
    danhSachPhanNhom: {
      mucDo: string;
      tenDoiTuong: string;
      idDoiTuong: string;
      nhomVaiTroId: {
        _id: string;
        danhSachChucNang: string[];
      };
    }[];
    vaiTro: string;
  };
}> {
  const fetchUserInfo: () => Promise<any> = async () => {
    try {
      const auth = localStorage.getItem('vaiTro') as ESystemRole;
      const token = localStorage.getItem('token');
      let currentUser;

      if (auth && token) {
        if ([ESystemRole.Admin, ESystemRole.QuanTriVien].includes(auth))
          currentUser = (await getInfoAdmin())?.data;
        else currentUser = (await getInfo())?.data;
      }
      return currentUser;
    } catch (error) {
      const { location } = history;
      if (!pathAuth.includes(location.pathname)) history.push(loginPath);
    }
    return undefined;
  };
  if (history.location.pathname !== loginPath) {
    const currentUser: Login.Profile & Login.ProfileAdmin = await fetchUserInfo();
    // const phanNhom = await getPhanNhom();
    return {
      fetchUserInfo,
      currentUser,
      settings: {
        primaryColor: 'daybreak',
        layout: 'side',
      },
      authorizedRoles: [],
      // phanNhom,
    };
  }

  return {
    fetchUserInfo,
    settings: { primaryColor: 'daybreak', layout: 'side' },
  };
}

const authHeaderInterceptor = (url: string, options: RequestOptionsInit) => {
  const token = localStorage.getItem('token');
  const authHeader = { ...(token && { Authorization: `Bearer ${token}` }) };
  return {
    url: `${url}`,
    options: { ...options, interceptors: true, headers: authHeader },
  };
};

/**
 * @see https://beta-pro.ant.design/docs/request-cn
 */
export const request: RequestConfig = {
  errorHandler: (error: ResponseError) => {
    const { messages } = getIntl(getLocale());
    const { response } = error;

    if (response && response.status) {
      const { status, statusText, url } = response;
      const requestErrorMessage = messages['app.request.error'];
      const errorMessage = `${requestErrorMessage} ${status}: ${url}`;
      const errorDescription = messages[`app.request.${status}`] || statusText;
      notification.error({
        message: errorMessage,
        description: errorDescription,
      });
    }

    if (!response) {
      notification.error({
        description: 'Yêu cầu gặp lỗi',
        message: 'Bạn hãy thử lại sau',
      });
    }
    throw error;
  },
  requestInterceptors: [authHeaderInterceptor],
};

// ProLayout  https://procomponents.ant.design/components/layout
export const layout: RunTimeLayoutConfig = ({ initialState }) => {
  return {
    unAccessible: <NotAccessible />,
    noFound: <NotFoundContent />,
    rightContentRender: () => <RightContent marginTopGioiThieuChung={17} />,
    disableContentMargin: false,
    waterMarkProps: {
      content: initialState?.currentUser?.username,
    },
    // headerRender: (props, dom) => <div style={{ backgroundColor: '#CC0D00' }}>{dom}</div>,
    isMobile: true,
    // footerRender: () => <GlobalFooter />,
    onPageChange: () => {
      const { location } = history;
      const token = localStorage.getItem('token');
      const vaiTro = localStorage.getItem("vaiTro");
      if (!token && location.pathname !== loginPath && !pathAuth.includes(location.pathname)) {
        history.push(loginPath);
      } else if (token) {
        if (
          (vaiTro == ESystemRole.User || vaiTro == ESystemRole.Admin) &&
          location.pathname === loginPath
        ) {
          console.log('vaiTro', data.path[`${vaiTro || initialState?.currentUser?.systemRole}`]);
          history.push(data.path[`${vaiTro || initialState?.currentUser?.systemRole}`]);
        } else {
          history.push(location.pathname);
        }
      }
    },
    logo: <img src="/favicon.ico" onClick={() => history.push('/')} />,
    menuItemRender: (item, dom) => {
      return (
        <Tooltip
          placement={
            initialState?.currentUser?.systemRole === ESystemRole.ThiSinh ? 'bottom' : 'right'
          }
          title={item.name}
        >
          <div
            style={{ flex: 'auto', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}
            onClick={() => {
              history.push(item?.path ?? '/');
            }}
          >
            {dom}
          </div>
        </Tooltip>
      );
    },
    childrenRender: (dom) => {
      return <ErrorBoundary>{dom}</ErrorBoundary>;
    },
    menuHeaderRender: undefined,
    ...initialState?.settings,
    title: 'Chatbot',
  };
};
