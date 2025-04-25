import { answer } from '@/models/answer';
import { intent } from '@/services/Intent/intent';
import { data } from '@/utils/data';
import { avatar } from '@/assets/logo.png';
import { message } from 'antd';
// import Service from '@/pages/QuanLyNguoiDung/service';
import { ActionType } from '@ant-design/pro-table';
import { useRef, useState } from 'react';
import question from '@/services/Question/question';
// import baseModel from './baseModel';

export interface IQuestionRecord {
  id: string;
  question: string;
  answer: string;
  is_active: boolean;
}

export default () => {
  const [danhSach, setDanhSach] = useState<IQuestionRecord[]>([]);
  const [showDrawer, setShowDrawer] = useState(false);
  const [edit, setEdit] = useState(false);
  const [loading, setLoading] = useState(false);
  const [record, setRecord] = useState<IQuestionRecord | {}>({});
  const [visibleForm, setVisibleForm] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [filterInfo, setFilterInfo] = useState<any>({});
  const [condition, setCondition] = useState<any>({});
  const [visible, setVisible] = useState<boolean>(false);

  const getData = async (IntentId: any) => {
    try {
      setLoading(true);
      const res = await question.get({
        "filters[intent][$eq]": IntentId?.intent_id,
        "pagination[page]": page,
        "pagination[pageSize]": limit,
        ...condition,
      });
      // console.log(res);
      if (res.status === 200) {
        setDanhSach(res.data?.data?.map((item: any, index: number) => {
          return {
            id: item?.id,
            question: item?.attributes?.question,
            answer: item?.attributes?.answer,
            is_active: item?.attributes?.is_active,
          }
        }));
        setTotal(res.data?.meta?.pagination?.total);
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
      const res = await question.get_by_id(id);
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
      const res = await question.add(data);
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
  const addFile = async (data: any) => {
    try {
      setLoading(true);
      const res = await question.addFile(data);
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
  const addFileQuestion = async (data: any) => {
    try {
      setLoading(true);
      const res = await question.addFileQuestion(data);
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
      const res = await question.upd(data);
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
      const res = await question.del(id);
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
    question,

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
    addFile,
    addFileQuestion,
  };
};
