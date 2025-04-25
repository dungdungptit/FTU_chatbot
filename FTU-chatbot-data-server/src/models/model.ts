import { data } from '@/utils/data';
import { avatar } from '@/assets/logo.png';
import { message } from 'antd';
// import Service from '@/pages/QuanLyNguoiDung/service';
import { ActionType } from '@ant-design/pro-table';
import { useRef, useState } from 'react';
import model from '@/services/Model/model';
import { set } from 'lodash';
// import baseModel from './baseModel';

export interface IModelRecord {
  id: string;
  model_name: string;
  isActivate: boolean;
}

export default () => {
  const [danhSach, setDanhSach] = useState<IModelRecord[]>([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [edit, setEdit] = useState(false);
  const [loading, setLoading] = useState(false);
  const [record, setRecord] = useState<IModelRecord | {}>({});
  const [visibleForm, setVisibleForm] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [filterInfo, setFilterInfo] = useState<any>({});
  const [condition, setCondition] = useState<any>({});
  const [visible, setVisible] = useState<boolean>(false);
  const [loadingTrain, setLoadingTrain] = useState<boolean>(false);
  const [percent, setPercent] = useState<number>(0);
  const [status, setStatus] = useState<string>('active');
  const [isClickTrain, setIsClickTrain] = useState<boolean>(false);

  const getData = async () => {
    try {
      setLoading(true);
      console.log(page, limit, condition);
      const res = await model.get({
        page,
        size: limit,
        ...condition,
      });
      console.log(res);
      if (res.status === 200) {
        setDanhSach(res.data);
        setTotal(res.data.length || 0);
      }
    } catch (error) {
      setDanhSach([]);
      setTotal(0);
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const getDataById = async (id: string) => {
    try {
      setLoading(true);
      const res = await model.get_by_id(id);
      console.log(res);
      if (res.status === 200) {
        setRecord(res.data);
      }
    } catch (error) {
      setRecord({});
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const add = async (data: any) => {
    try {
      setLoading(true);
      const res = await model.add(data);
      console.log(res);
      if (res.status === 200) {
        message.success('Thêm mới thành công');
        setVisibleForm(false);
        getData();
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };
  const activate = async (data: any) => {
    try {
      setLoading(true);
      const res = await model.activate(data);
      console.log(res);
      if (res.status === 200) {
        message.success('Đã kích hoạt thành công');
        getData();
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const autoTrain = async () => {
    try {
      setLoadingTrain(true);
      setIsClickTrain(true);
      setPercent(0);
      setStatus('active');
      setTimeout(() => {
        setPercent(20);
      }, 1000);
      setTimeout(() => {
        setPercent(50);
      }, 1500);
      setTimeout(() => {
        setPercent(80);
      }, 3500);
      setTimeout(() => {
        setPercent(90);
      }, 5000);
      // await model.exportnludata('nlu');
      const token = localStorage.getItem('token');
      if (token) {
        const res = await model.autotrain(token);
        console.log(res);
        if (res.status === 200) {
          message.success('Cập nhập dữ liệu mới thành công, hãy trò chuyện với chatbot để kiểm tra lại', 10);
          setPercent(100);
          setStatus('success');
        }
      }
      else {
        message.error('Lỗi thực hiện');
        setPercent(100);
        setStatus('exception');
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
      setPercent(100);
      setStatus('exception');
    } finally {
      setLoadingTrain(false);
    }
  };

  const upd = async (data: any) => {
    try {
      setLoading(true);
      const res = await model.upd(data);
      if (res.status === 200) {
        message.success('Cập nhật thành công');
        setVisibleForm(false);
        getData();
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const del = async (id: string) => {
    try {
      setLoading(true);
      const res = await model.del(id);
      if (res.status === 200) {
        message.success('Xóa thành công');
        getData();
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  return {
    model,

    condition,
    danhSach,
    edit,
    filterInfo,
    loading,
    limit,
    total,
    page,
    record,
    visible,
    visibleForm,
    showDrawer,

    setCondition,
    setDanhSach,
    setEdit,
    setFilterInfo,
    setLimit,
    setTotal,
    setPage,
    setRecord,
    setVisible,
    setVisibleForm,
    setShowDrawer,

    getDataById,

    getData,
    add,
    upd,
    del,
    autoTrain,
    loadingTrain,
    percent,
    status,
    isClickTrain,
    activate
  };
};
