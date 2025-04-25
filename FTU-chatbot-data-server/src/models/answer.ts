import { intent } from '@/services/Intent/intent';
import { data } from '@/utils/data';
import { avatar } from '@/assets/logo.png';
import { message } from 'antd';
// import Service from '@/pages/QuanLyNguoiDung/service';
import { ActionType } from '@ant-design/pro-table';
import { useRef, useState } from 'react';
import answer from '@/services/Answer/answer';
// import baseModel from './baseModel';

export interface IAnswerRecord {
  id: string;
  answer: string;
}

export default () => {
  const [danhSach, setDanhSach] = useState<IAnswerRecord[]>([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [edit, setEdit] = useState(false);
  const [loading, setLoading] = useState(false);
  const [record, setRecord] = useState<IAnswerRecord | {}>({});
  const [visibleForm, setVisibleForm] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [filterInfo, setFilterInfo] = useState<any>({});
  const [condition, setCondition] = useState<any>({});
  const [visible, setVisible] = useState<boolean>(false);
  const [isImage, setIsImage] = useState<boolean>(false);

  const getData = async (IntentId: any) => {
    try {
      setLoading(true);
      const res = await answer.get({
        intent_id: IntentId?.intent_id,
        page,
        size: limit,
        ...condition,
      });
      if (res.status === 200) {
        res.data = res.data?.sort((a: any, b: any) => a?.sort - b?.sort);
        setDanhSach(res.data.map((item: any) => ({ ...item, image: item?.answer?.Image || '', answer: item?.answer?.answer || '' })) || []);
        setTotal(res.data?.length || 0);
      }
      console.log(danhSach);
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
      const res = await answer.get_by_id(id);
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
      const res = await answer.add(data);
      console.log(res);
      if (res.status === 200) {
        message.success('Thêm mới thành công');
        setVisibleForm(false);
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const addWithImage = async (data: any) => {
    try {
      setLoading(true);
      const res = await answer.addWithImage(data);
      // console.log(res);
      if (res.status === 200) {
        message.success('Thêm mới thành công');
        setVisibleForm(false);
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const addFile = async (data: any) => {
    try {
      setLoading(true);
      const res = await answer.addFile(data);
      console.log(res);
      if (res.status === 200) {
        message.success('Thêm mới thành công');
        setVisibleForm(false);
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const upd = async (data: any) => {
    try {
      setLoading(true);
      const res = await answer.upd(data);
      if (res.status === 200) {
        message.success('Cập nhật thành công');
        setVisibleForm(false);
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const updWithImage = async (data: any) => {
    try {
      setLoading(true);
      const res = await answer.updWithImage(data);
      if (res.status === 200) {
        message.success('Cập nhật thành công');
        setVisibleForm(false);
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  const del = async (id: string, intent_id: string) => {
    try {
      setLoading(true);
      const res = await answer.del(id);
      if (res.status === 200) {
        message.success('Xóa thành công');
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };
  const delWithImage = async (id: string, intent_id: string) => {
    try {
      setLoading(true);
      const res = await answer.delWithImage(id);
      if (res.status === 200) {
        message.success('Xóa thành công');
      }
    } catch (error) {
      message.error('Lỗi thực hiện');
    } finally {
      setLoading(false);
    }
  };

  return {
    answer,

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
    addWithImage,
    upd,
    updWithImage,
    del,
    isImage,
    setIsImage,
    addFile,
    delWithImage
  };
};
